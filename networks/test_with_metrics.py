
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

torch.manual_seed(1)
device = torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    device = torch.device("cuda")

character_set = "-0123456789#"  # space is for nothing
# create model
model = model_list.ocrnet(device=device, num_classes=len(character_set))
model.features = torch.nn.DataParallel(model.features)

checkpoint = torch.load('checkpoint.pth.tar', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.to(device)

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
        if c == 100:
            continue
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

    for i, (input, target, emb_len, target_len) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        start = time.time()

        output = model(input_var)
        #output = softmax(output.detach().cpu().numpy(), axis=2)
        output = torch.nn.functional.log_softmax(output, 2)
        output = np.argmax(output.detach().cpu().numpy(), axis=2)

        label_from_model = arr_to_label(output[0])

        #cut repeating symbols
        res = ''
        for c in label_from_model:
            if len(res) == 0:
                res = res + c
            elif c != res[-1]:
                res = res + c
        label_from_model = res
        res = ''
        for c in label_from_model:
            if c != '-':
                res = res + c
        label_from_model = res

        label_true = arr_to_label(target.numpy()[0])

        print(label_from_model)
        print(label_true)

        prc = round(SQ(None, label_from_model, label_true).ratio() * 100, 2)
        all_percent_coincidence = all_percent_coincidence + prc
        print(f"Percent coincidence: {prc}%")

        print("time: ", time.time() - start)

    print('all_percent_coincidence = ', all_percent_coincidence / val_loader.__len__())

validate(val_loader, model)