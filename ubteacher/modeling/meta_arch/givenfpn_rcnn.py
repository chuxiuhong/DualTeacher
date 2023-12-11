# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
import torch

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
            self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
            if given_proposals is not None:
                for pesudo_rpn in given_proposals:
                    if len(pesudo_rpn) == len(proposals_rpn):
                        for i in range(len(proposals_rpn)):
                            boxes = Boxes(torch.cat([proposals_rpn[i].get("proposal_boxes").tensor,pesudo_rpn[i].get("gt_boxes").tensor]))
                            objectness_logits = torch.cat([proposals_rpn[i].get("objectness_logits"),pesudo_rpn[i].get("objectness_logits")])
                            fields = proposals_rpn[i].get_fields()
                            fields["proposal_boxes"] = boxes
                            fields["objectness_logits"] = objectness_logits

            for i in range(len(proposals_rpn)):
                objectness_logits, indices = proposals_rpn[i].get("objectness_logits").sort(descending=True)
                boxes = proposals_rpn[i].get("proposal_boxes").tensor.index_select(0, indices)
                fields = proposals_rpn[i].get_fields()
                fields["proposal_boxes"] = Boxes(boxes)
                fields["objectness_logits"] = objectness_logits

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], features

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return features, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None
        elif branch == "only_feature":
            return features
    # def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)

    #     if detected_instances is None:
    #         if self.proposal_generator:
    #             proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]

    #         results, _ = self.roi_heads(images, features, proposals, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(
    #             features, detected_instances
    #         )

    #     if do_postprocess:
    #         return GeneralizedRCNN._postprocess(
    #             results, batched_inputs, images.image_sizes
    #         )
    #     else:
    #         return results
