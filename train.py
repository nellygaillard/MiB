import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from functools import reduce

from utils.loss import KnowledgeDistillationLoss, BCEWithLogitsLossWithIgnoreIndex, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, IcarlLoss
from utils import get_regularizer


class Trainer:
    def __init__(self, model, model_old, opts, trainer_state=None, classes=None):

        self.model_old = model_old
        self.model = model
        self.step = opts.step

        # for pseudo labeling
        self.threshold = opts.threshold


        if classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = tot_classes - new_classes
        else:
            self.old_classes = 0

        # Select the Loss Type
        reduction = 'none'

        self.supervision_losses = opts.supervl          # if to add the bisenet supervision losses
        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(old_cl=self.old_classes, ignore_index=255, reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        # ILTSS
        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and model_old is not None
        self.lde_loss = nn.MSELoss()

        self.lkd = opts.loss_kd
        self.lkd_flag = self.lkd > 0. and model_old is not None
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=opts.alpha)
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

        # Regularization
        regularizer_state = trainer_state['regularizer'] if trainer_state is not None else None
        self.regularizer = get_regularizer(model, model_old, opts, regularizer_state)
        self.regularizer_flag = self.regularizer is not None
        self.reg_importance = opts.reg_importance

        self.ret_intermediate = self.lde

    def entropy(probabilities):
        """
        Computes the entropy per pixel.
        :param probabilities: Tensor of shape (b, c, w, h).
        :return: One entropy per pixel, shape (b, w, h)
        """
        factor = 1 / math.log(probabilities.shape[1] + 1e-8)
        return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)     # viene calcolata la media sul batch?


    def train(self, cur_epoch, optim, train_loader, scheduler=None, print_int=10, logger=None):
        """Train and return epoch loss"""
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        model = self.model
        criterion = self.criterion

        scaler = GradScaler()

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        #train_loader.sampler.set_epoch(cur_epoch)

        model.train()
        for cur_step, (images, labels) in enumerate(train_loader):

            images = images.cuda()
            labels = labels.cuda().long()
            
            original_labels = labels.clone()
            
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)

            if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag) and self.model_old is not None:
                with torch.no_grad():
                    with autocast():
                        outputs_old, empty_dict = self.model_old(images)
                        
            #aggiunto da Andrea per pseudo-labeling
           
            if self.step > 0:
                if opts.pseudo == "naive":
                    mask_background = labels < self.old_classes     # seleziono solo i labels non corrispondenti alle nuove classi
                    labels[mask_background] = outputs_old.argmax(dim=1)[mask_background]
                elif opts.pseudo == "entropy":
                    mask_background = labels < self.old_classes
                    probs = torch.softmax(outputs_old, dim=1)           # BxCxWxH
                    max_probs, pseudo_labels = probs.max(dim=1)         # BxWxH
                    #mask_valid_pseudo = (entropy(probs) / self.max_entropy) < self.thresholds[pseudo_labels]
                    mask_valid_pseudo = entropy(probs) < self.thresholds
                    labels[~mask_valid_pseudo & mask_background] = 255          # put to 255 (?) pixels for which the model is uncertain
                    labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo & mask_background] # use the pseudo-labels for pixels for which the model is confident enough

            #######################################

            optim.zero_grad()
            with autocast():
                outputs, output_sup1, output_sup2 = model(images)

            # xxx BCE / Cross Entropy Loss
            with autocast():
                if self.supervision_losses:
                    loss2 = criterion(output_sup1, labels)
                    loss3 = criterion(output_sup2, labels)
                    loss_supervision = loss2 + loss3
                if not self.icarl_only_dist:
                    loss = criterion(outputs, labels)  # B x H x W
                else:
                    loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                if self.supervision_losses:
                    loss += loss_supervision

                loss = loss.mean()  # scalar

                if self.icarl_combined:
                    # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                    n_cl_old = outputs_old.shape[1]
                    # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                    l_icarl = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                                  torch.sigmoid(outputs_old))

                # xxx ILTSS (distillation on features or logits)
                if self.lde_flag:
                    lde = self.lde * self.lde_loss(output_sup1, features_old['body'])

                if self.lkd_flag:
                    # resize new output to remove new logits and keep only the old ones
                    lkd = self.lkd * self.lkd_loss(outputs, outputs_old)

                # xxx first backprop of previous loss (compute the gradients for regularization methods)
                loss_tot = loss + lkd + lde + l_icarl

            # Scales loss. Calls backwards() on scaled loss to create scaled gradients
            scaler.scale(loss_tot).backward()

            # xxx Regularizer (EWC, RW, PI) ------- not needed for ilss project
            if self.regularizer_flag:
                if distributed.get_rank() == 0:
                    self.regularizer.update()
                l_reg = self.reg_importance * self.regularizer.penalty()
                if l_reg != 0.:
                    with torch.cuda.amp.autocast():
                        l_reg.backward()

            #optim.step()
            scaler.step(optim)

            # updates the scale for next iteration
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                            f" Loss={interval_loss}")
                logger.debug(f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}")
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss', interval_loss, x)
                interval_loss = 0.0

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).cuda()
        reg_loss = torch.tensor(reg_loss).cuda()

        # reduce the tensor data across all machines: not needed in our case, only the process with dst
        # is going to receive the final result
        #torch.distributed.reduce(epoch_loss, dst=0)
        #torch.distributed.reduce(reg_loss, dst=0)


        epoch_loss = epoch_loss / len(train_loader) # !!!
        reg_loss = reg_loss / len(train_loader)     # !!!

        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")

        return (epoch_loss, reg_loss)

    def validate(self, loader, metrics, ret_samples_ids=None, logger=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        #device = self.device
        criterion = self.criterion
        model.eval()

        scaler = GradScaler()

        class_loss = 0.0
        reg_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        ret_samples = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):

                images = images.cuda()
                labels = labels.cuda().long()

                if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag) and self.model_old is not None:
                    with torch.no_grad():
                        with autocast():
                            outputs_old, dictionary = self.model_old(images)

                with autocast():
                    outputs, dictionary = model(images)


                    # xxx BCE / Cross Entropy Loss
                    if not self.icarl_only_dist:
                        loss = criterion(outputs, labels)  # B x H x W
                    else:
                        loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                    loss = loss.mean()  # scalar

                    if self.icarl_combined:
                        # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                        n_cl_old = outputs_old.shape[1]
                        # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                        l_icarl = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                                      torch.sigmoid(outputs_old))

                    # xxx ILTSS (distillation on features or logits)
                    if self.lde_flag:
                        lde = self.lde_loss(features['body'], features_old['body'])

                    if self.lkd_flag:
                        lkd = self.lkd_loss(outputs, outputs_old)

                    # xxx Regularizer (EWC, RW, PI)
                    if self.regularizer_flag:
                        l_reg = self.regularizer.penalty()

                    class_loss += loss.item()
                    reg_loss += l_reg.item() if l_reg != 0. else 0.
                    reg_loss += lkd.item() + lde.item() + l_icarl.item()

                _, prediction = outputs.max(dim=1)

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((images[0].detach().cpu().numpy(),
                                        labels[0],
                                        prediction[0]))

            # collect statistics from multiple processes
            #metrics.synch(device)
            score = metrics.get_results()

            class_loss = torch.tensor(class_loss).cuda()
            reg_loss = torch.tensor(reg_loss).cuda()

            class_loss = class_loss / len(loader)   # !!
            reg_loss = reg_loss / len(loader)       # !!

            if logger is not None:
                logger.info(f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)")

        return (class_loss, reg_loss), score, ret_samples

    def state_dict(self):
        state = {"regularizer": self.regularizer.state_dict() if self.regularizer_flag else None}

        return state

    def load_state_dict(self, state):
        if state["regularizer"] is not None and self.regularizer is not None:
            self.regularizer.load_state_dict(state["regularizer"])
