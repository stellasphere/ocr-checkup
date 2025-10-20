from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo
from ocrcheckup.cost import CostType, ModelCost

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info
from .consts import OCR_VLM_PROMPT

class Qwen_2_5_VL_7B(OCRBaseModel):
    def info(self) -> OCRModelInfo:
        return OCRModelInfo(
            name="Qwen2.5-VL",
            version="Qwen2.5-VL-7B-Instruct",
            tags=["local", "lmm"], 
            cost_type="compute"
        )

    def __init__(self):
        model_info = self.info()
        self.model_id = model_info.version
        super().__init__(model_id=self.model_id)

        print(f"Qwen2.5-VL initializing with checkpoint: {self.model_id}")

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                trust_remote_code=True 
            )


        except Exception as e:
            print(f"Error loading Qwen 2.5 VL model or processor: {e}")
            raise

    def evaluate(self, image) -> OCRModelResponse:
        # 1. Ensure image is PIL format
        if not isinstance(image, Image.Image):
             image_pil = Image.fromarray(image)
        else:
             image_pil = image

        # 2. Prepare messages list including the actual image object
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": OCR_VLM_PROMPT},
                ],
            }
        ]

        # 3. Apply chat template and process inputs using the utility
        try:
            # Get the text prompt part
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Use the helper function to prepare vision inputs
            image_inputs, video_inputs = process_vision_info(messages)

            # Process text and vision inputs together
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True
            )

            # Explicitly move input tensors to the model's device
            target_device = next(self.model.parameters()).device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

        except Exception as e:
             print(f"Error during Qwen processing: {e}")
             return OCRModelResponse(prediction="", success=False, error_message=str(e))


        # 4. Generate text using the model
        prediction = ""
        try:
            with torch.inference_mode(): # Use inference mode for efficiency
                 # Generate, assuming inputs are correctly placed by processor/device_map
                 generated_ids = self.model.generate(**inputs, max_new_tokens=512) # Increased tokens slightly for OCR

            # 5. Decode the generated IDs
            # Trim the input IDs from the generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            prediction = output_text[0].strip() if output_text else ""

        except Exception as e:
             print(f"Error during Qwen generation or decoding: {e}")
             # Return failure but include partial prediction if available (though likely empty)
             return OCRModelResponse(prediction=prediction, success=False, error_message=str(e))

        # 6. Return the response
        return OCRModelResponse(
            prediction=prediction
        ) 