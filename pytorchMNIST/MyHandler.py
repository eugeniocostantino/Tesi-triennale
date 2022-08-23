import io

import torch
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image

class MyHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.transform = transforms.ToTensor()

    def single_preprocess(self, r):
        image = r.get('data') or r.get('body')
        print(image)
        image = Image.open(io.BytesIO(image))
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image


    def preprocess(self, req):
        images = [self.single_preprocess(r) for r in req]
        batch = torch.cat(images)
        return batch

    def inference(self, input):
        output = self.model.forward(input)
        output = torch.nn.functional.softmax(output,1)
        return torch.argmax(output, 1)

    def postprocess(self, outputs):
        response = []
        for out in outputs.tolist():
            response.append({'prediction' : out})

        return response
