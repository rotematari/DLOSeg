import torch
import torch.nn as nn
from sam2_realtime.sam2.build_sam import build_sam2
from sam2_realtime.sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from dloseg.models.adapters import PromptEncoder,Classifier

class DLOSeg(nn.Module):
        def __init__(self, configs,device='cpu'):
            super().__init__()
            self.configs = configs
            # SAM2-RT model configs
            checkpoint_path = configs.get("checkpoint_path", '/home/admina/segmetation/DLOSeg/src/sam2_realtime/checkpoints/sam2.1_hiera_small.pt')
            model_cfg = configs.get("model_cfg", 'configs/sam2.1/sam2.1_hiera_s.yaml')
            self.multimask_output = configs.get("multimask_output", True)
            self.mask_threshold = configs.get("mask_threshold", 0.0)
            
            # CLIPSeg model configs
            self.promts = configs.get("promts", ['a wire'])
            
            
            # SAM2-RT model
            self.device = device
            self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
            self.predictor = SAM2ImagePredictor(self.sam2_model)

            # CLIPSeg model
            self.clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
            self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
            
            # Adapter
            self.adapter = PromptEncoder(configs)
            self.classifier = Classifier(configs)
            
        def clipseg_forward(self, image):
            """
            Run CLIPSeg model forward pass
            output: the output of the 2'nd to last attention layer of the CLIP model
            
            """
            inputs = self.processor(text=self.prompts,
                                    images=[image] * len(self.prompts),
                                    padding="max_length",
                                    return_tensors="pt",)
            # predict
            with torch.no_grad():
                outputs = self.clipseg_model(output_hidden_states = True,**inputs)
            
            return outputs.decoder_output.hidden_states[-2]
        
        def sam2_forward(self, image,return_logits=False):
            """
            Run SAM2 model forward pass with the adapter
            output: mask logits
            """
            
            self.predictor.set_image(image)
            
            # prompt encoder
            dpe = self.sam2_model.get_dense_pe()
            clipseg_embedding = self.clipseg_forward(image)
            
            # Adapter
            point_embeddings, embeddings_sampled = self.adapter(clipseg_embedding,dpe)
            
            # Predict masks
            batched_mode = None
            high_res_features = [
                feat_level[-1].unsqueeze(0)
                for feat_level in self.predictor._features["high_res_feats"]
            ]
            low_res_masks, iou_predictions, _, _ = self.predictor.model.sam_mask_decoder(
                image_embeddings=self._features["image_embed"][-1].unsqueeze(0),
                image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=None,
                dense_prompt_embeddings=point_embeddings,
                multimask_output=self.multimask_output,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            

            # Upscale the masks to the original image resolution
            masks = self.predictor._transforms.postprocess_masks(
                low_res_masks, self._orig_hw[-1]
            )
            if not return_logits:
                masks = masks > self.mask_threshold

            return masks, iou_predictions, low_res_masks ,embeddings_sampled
        
        def classifer_forward(self,point_embeddings,mask_tokens):
            """
            Run the classifier forward pass
            output: point embeddings, mask tokens
            """
            return self.classifier(point_embeddings,mask_tokens)
        