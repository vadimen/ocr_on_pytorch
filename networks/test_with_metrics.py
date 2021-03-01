
import os
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import model_list
import sys
from scipy.special import softmax

cwd = os.getcwd()
sys.path.append(cwd+'/../')
import datasets as datasets
from torchvision import transforms
import numpy as np

character_set = "0123456789# "  # space is for nothing
# create model
model = model_list.alexnet(num_classes=len(character_set))
model.features = torch.nn.DataParallel(model.features)

checkpoint = torch.load('checkpoint_v1_1.pth.tar', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

validation_dataset = datasets.MyDataset(
    img_dir='../test_data/test/',
    transform=transforms.Compose([
        #transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]), character_set=character_set)

batch_size = 1
val_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=batch_size, shuffle=True,
    num_workers=0, pin_memory=True)

def arr_to_label(arr):
    label = ''
    for c in arr:
        label = label + character_set[c]
    return label.strip()


from difflib import SequenceMatcher as SQ

try:
    from PIL import Image
except ImportError:
    import Image

def validate(val_loader, model):

    # switch to evaluate mode
    model.eval()
    all_percent_coincidence = 0

    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        start = time.time()

        output = model(input_var)
        output = softmax(output.detach().numpy(), axis=2)
        output = np.argmax(output, axis=2)

        print(output[0])
        print(target.numpy()[0])

        prc = round(SQ(None, arr_to_label(target.numpy()[0]), arr_to_label(output[0])).ratio() * 100, 2)
        all_percent_coincidence = all_percent_coincidence + prc
        print(f"Percent coincidence: {prc}%")

        print("time: ", time.time() - start)

    print('all_percent_coincidence = ', all_percent_coincidence / val_loader.__len__())

validate(val_loader, model)