import pathlib
import json
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.models import vgg16
from torchvision.transforms.functional import gaussian_blur
from PIL import Image
from skimage.io import imread

path = pathlib.Path(__file__).parent

with open(path / "classes.json", "r") as file:
    classe_indexes = json.load(file)
classes = {k: v for v, k in enumerate(classe_indexes)}

device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
turtle  = torch.tensor(imread(path / "turtle.png"), device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
model = vgg16(pretrained=True)
model.eval()
model.to(device)
normalize = transforms.Compose([lambda x: x/255.,
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
unnormalize = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),                 
                                  transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),                                                     
                                  lambda x: x*255])
start_size, end_size, n_steps = (32, 32), (224, 224), 100
image = torch.rand((1, 3, *start_size), dtype=torch.float32, device=device) * 6 - 3
# image = F.interpolate(normalize(turtle), size=start_size, mode="bilinear")
images = []

for step in range(1, n_steps+1):
    param = torch.nn.parameter.Parameter(image)
    optimizer = torch.optim.Adam([param], lr=1.0E-3, weight_decay=0.)
    for k in range(10):
        optimizer.zero_grad()
        upscaled = F.interpolate(param, size=end_size, mode="bilinear")
        array = unnormalize(upscaled.detach().cpu()).permute(0, 2, 3, 1).squeeze(0).numpy()
        inf, sup = array.min(), array.max()
        array = np.clip(array, 0, 255).astype("uint8")
        images.append(array)
        p = torch.softmax(model(upscaled), dim=-1)[:, 36]
        loss = -torch.log(p).mean()
        loss.backward()
        optimizer.step()
        max_grad = param.grad.max()
        print(f"step {step*10 + k}: p = {p[0]:.3g}, loss = {loss.item():.3g}, max grad = {max_grad:.3g}, inf/sup = ({inf:.3g}, {sup:.3g})")
    image = F.interpolate(image, size=tuple(int(round(i + (s-i)*(step+1)/n_steps)) for i, s in zip(start_size, end_size)), mode="bilinear")

start_image = Image.fromarray(images[0]).convert('P')
other_images = [Image.fromarray(im).convert('P') for im in images[1:]]
start_image.save(path / 'deep_dreaming.webp', save_all=True, append_images=other_images, loop=0, duration=10, transparency=0, disposal=2)







# import requests
# from io import BytesIO
# import matplotlib.pyplot as plt

# #Class to register a hook on the target layer (used to get the output channels of the layer)
# class Hook():
#     def __init__(self, module, backward=False):
#         if backward==False:
#             self.hook = module.register_forward_hook(self.hook_fn)
#         else:
#             self.hook = module.register_backward_hook(self.hook_fn)
#     def hook_fn(self, module, input, output):
#         self.input = input
#         self.output = output
#     def close(self):
#         self.hook.remove()
  
# #Function to make gradients calculations from the output channels of the target layer  
# def get_gradients(net_in, net, layer):     
#   net_in = net_in.unsqueeze(0).cuda()
#   net_in.requires_grad = True
#   net.zero_grad()
#   hook = Hook(layer)
#   net_out = net(net_in)
#   loss = hook.output[0].norm()
#   loss.backward()
#   return net_in.grad.data.squeeze()

# #denormalization image transform
# denorm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),                 
#                               transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),                                                     
#                               ])

# #Function to run the dream.
# def dream(image, net, layer, iterations, lr):
#   image_tensor = transforms.ToTensor()(image)
#   image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor).cuda()
#   for i in range(iterations):
#     gradients = get_gradients(image_tensor, net, layer)
#     image_tensor.data = image_tensor.data + lr * gradients.data

#   img_out = image_tensor.detach().cpu()
#   img_out = denorm(img_out)
#   img_out_np = img_out.numpy().transpose(1,2,0)
#   img_out_np = np.clip(img_out_np, 0, 1)
#   img_out_pil = Image.fromarray(np.uint8(img_out_np * 255))
#   return img_out_pil

# #Input image
# url = 'https://s3.amazonaws.com/pbblogassets/uploads/2018/10/22074923/pink-sky-cover.jpg'
# response = requests.get(url)
# img = Image.open(BytesIO(response.content))
# orig_size = np.array(img.size)
# new_size = np.array(img.size)*0.5
# img = img.resize(new_size.astype(int))
# layer = list(model.features.modules() )[27]

# img = dream(img, model, layer, 20, 1)

# img = img.resize(orig_size)
# fig = plt.figure(figsize = (10 , 10))
# plt.imshow(img)
# plt.show()
