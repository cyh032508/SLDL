import json

file_path = 'cnn_v2-stn.ipynb'

with open(file_path, 'r') as f:
    nb = json.load(f)

def find_cell_index(nb, content_snippet):
    for i, cell in enumerate(nb['cells']):
        if 'source' in cell:
            source = ''.join(cell['source'])
            if content_snippet in source:
                return i
    return -1

# 1. Data Augmentation
# Find the code cell with exactly "# TODO"
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']).strip()
        if source == '# TODO':
            cell['source'] = [
                "# TODO\n",
                "rotater = v2.RandomRotation(degrees=(0, 180))\n",
                "rotated_imgs = [rotater(img) for _ in range(4)]\n",
                "plot([img] + rotated_imgs)\n",
                "\n",
                "grayscaler = v2.Grayscale()\n",
                "gray_img = grayscaler(img)\n",
                "plot([img, gray_img], cmap='gray')"
            ]
            print("Updated Data Augmentation cell")
            break

# 2. Q&A
idx2 = find_cell_index(nb, '1. images 的shape 是什麼?')
if idx2 != -1:
    nb['cells'][idx2]['source'] = [
        "### TODO\n",
        "\n",
        "回答下列問題\n",
        "1. images 的shape 是什麼? 各維度分別是什麼意思?\n",
        "   - `torch.Size([6, 3, 32, 32])`\n",
        "   - Batch Size (6), Channels (3), Height (32), Width (32)\n",
        "\n"
    ]
    print("Updated Q&A cell")

# 3. Model Definition
idx3 = find_cell_index(nb, 'net = torch.nn.Sequential')
if idx3 != -1:
    cell = nb['cells'][idx3]
    new_source = []
    for line in cell['source']:
        if '# TODO' in line:
            new_source.append("# TODO\n")
            new_source.append("    torch.nn.Conv2d(3, 6, 5),\n")
            new_source.append("    torch.nn.ReLU(),\n")
            new_source.append("    torch.nn.MaxPool2d(2, 2),\n")
            new_source.append("    torch.nn.Conv2d(6, 16, 5),\n")
            new_source.append("    torch.nn.ReLU(),\n")
            new_source.append("    torch.nn.MaxPool2d(2, 2),\n")
            new_source.append("    torch.nn.Flatten(),\n")
            new_source.append("    torch.nn.Linear(16 * 5 * 5, 120),\n")
            new_source.append("    torch.nn.ReLU(),\n")
            new_source.append("    torch.nn.Linear(120, 84),\n")
            new_source.append("    torch.nn.ReLU(),\n")
            new_source.append("    torch.nn.Linear(84, 10),\n")
        else:
            new_source.append(line)
    cell['source'] = new_source
    print("Updated Model Definition cell")

# 4. Prediction
idx4 = find_cell_index(nb, '# pass images through net')
if idx4 != -1:
    cell = nb['cells'][idx4]
    cell['source'] = [
        "# TODO\n",
        "# pass images through net and print predicted and true labels\n",
        "outputs = net(images)\n",
        "_, predicted = torch.max(outputs, 1)\n",
        "\n",
        "print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(6)))\n",
        "print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(6)))"
    ]
    print("Updated Prediction cell")

# 5. Accuracy
idx5 = find_cell_index(nb, '# Compute and print out test accuracy')
if idx5 != -1:
    cell = nb['cells'][idx5]
    cell['source'] = [
        "# TODO\n",
        "# Compute and print out test accuracy\n",
        "correct = 0\n",
        "total = 0\n",
        "with torch.no_grad():\n",
        "    for data in testloader:\n",
        "        images, labels = data[0].to(device), data[1].to(device)\n",
        "        outputs = net(images)\n",
        "        _, predicted = torch.max(outputs.data, 1)\n",
        "        total += labels.size(0)\n",
        "        correct += (predicted == labels).sum().item()\n",
        "\n",
        "print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')"
    ]
    print("Updated Accuracy cell")

with open(file_path, 'w') as f:
    json.dump(nb, f, indent=1)
print("Notebook updated successfully")
