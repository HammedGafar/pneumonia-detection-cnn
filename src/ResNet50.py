import torch
import torch.nn as nn
import torchvision
from torch import optim
import torch.nn.functional as F
from PIL import Image


import numpy as np
import cv2


import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets


from torchvision.models import densenet161
import torch.nn as nn
import torchvision
from torch import optim
import torch.nn.functional as F
from PIL import Image

import copy

import matplotlib.pyplot as plt

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets


from torchvision.models import resnet50


class ResNet50:
    def __init__(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.data_dir = 'chest_xray'
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
        # Create model with pretrained weights
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Linear(2048, 2)
        self.model = self.model.to(self.device)
        
        

        
        self.best_val_loss = float('inf')
        
        self.activation = {}
        
    def print_model(self):
        print(self.model)
        
    
    def check_frozen_layer(self):
        model = self.model
        print("CHECKING FROZEN AND UNFROZEN LAYERS")
        print("================================================================================")
        print()
        print()
        # Check for frozen layers
        for name, param in model.named_parameters():
            if param.requires_grad == False:
                print(f" {name} is frozen")
            else:
                print(f" {name} is unfrozen")

        print("================================================================================")
        
        
    def init_dataloader(self):

        generator = torch.Generator(device=self.device)
        generator.manual_seed(7)
        
        self.train_dataset = datasets.ImageFolder(f'{self.data_dir}/train', transform=self.transform)
        self.val_dataset   = datasets.ImageFolder(f'{self.data_dir}/val', transform=self.transform)
        self.test_dataset  = datasets.ImageFolder(f'{self.data_dir}/test', transform=self.transform)
        


        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader   = DataLoader(self.val_dataset, batch_size=32)
        self.test_loader  = DataLoader(self.test_dataset, batch_size=1, shuffle=True, generator=generator)
        
        
    def print_dataset_class_info(self):
        # Print dataset sizes
        print("\nDataset sizes:")
        print(f"Training set: {len(self.train_dataset)} images")
        print(f"Validation set: {len(self.val_dataset)} images")
        print(f"Test set: {len(self.test_dataset)} images")

        # Access class information
        print(f"Classes: {self.train_dataset.classes}")  # ['NORMAL', 'PNEUMONIA']
        print(f"Class mapping: {self.train_dataset.class_to_idx}")  # {'NORMAL': 0, 'PNEUMONIA': 1}
        
    def test_model(self,model_input, test_loader_input):

            model = model_input
            test_loader = test_loader_input

            test_loss = 0.0
            correct = 0
            total = 0

            # Set model to evaluation mode
            model.eval()

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Forward pass
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, labels)
                    test_loss += loss.item()

                    # (value, indices)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            # Calculate final metrics
            test_loss = test_loss / len(test_loader)
            test_accuracy = correct / total


            return test_loss, test_accuracy
        
        
    def eval_model(self, model_input, eval_loader_input):

        model = model_input
        eval_loader = eval_loader_input

        # Initialize variables for tracking performance

        model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)
                running_loss += loss.item()

                # Get predictions
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        # Calculate metrics
        eval_loss = running_loss / len(eval_loader)
        eval_accuracy = correct / total



        return eval_loss, eval_accuracy
    
    def train_model(self,model_input, train_loader_input, val_loader_input, optimizer_input, epochs=20):

        model = model_input
        train_loader = train_loader_input
        val_loader = val_loader_input
        optimizer = optimizer_input

        # Training history
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            
            # Training phase
            model.train()
            running_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_loader:
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                print(inputs.shape)
                print(labels.shape)
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Track metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Calculate epoch metrics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = train_correct / train_total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            # Validation phase
            val_loss, val_acc = self.eval_model(model, val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print("======================================================================================================")
                print("saving best model......")
                print("======================================================================================================")

                # Save relevant information
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }

                torch.save(checkpoint, 'best_model_full.pt')



            # Print epoch results
            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n')

        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
        
        
    def progressive_training_ResNet50(self,model_input, train_loader_input, val_loader_input):

        model = model_input
        train_loader = train_loader_input
        val_loader = val_loader_input
        
        
        # First freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        print("Unfreezing conv1...")
        model.conv1.requires_grad_(True)
        
        print("Unfreezing bn1...")
        model.bn1.requires_grad_(True)

        # Initial optimizer for conv1 and bn1
        optimizer = optim.SGD([
            {'params': model.conv1.parameters(), 'lr': 0.02},
            {'params': model.bn1.parameters(), 'lr': 0.02}
        ])
        
        
        # Train
        print()
        print()
        print("=======================================================================")
        print("Training starts")
        print("=======================================================================")
        self.train_model(model, train_loader_input, val_loader_input, optimizer, epochs=1)
        
        #Freeze all
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze only layer1
        print("Unfreezing layer1...")
        for param in model.layer1.parameters():
            param.requires_grad = True
            
        # Create optimizer for layer1
        optimizer = optim.SGD(
            model.layer1.parameters(), 
            lr=0.01
        )
        
        # Train
        print("=======================================================================")
        print("Training layer1...")
        print("=======================================================================")
        self.train_model(model, train_loader_input, val_loader_input, optimizer, epochs=5)
        
        #Freeze all layers again
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze only layer2
        print("Unfreezing layer2...")
        for param in model.layer2.parameters():
            param.requires_grad = True
            
        # Create optimizer for layer2
        optimizer = optim.SGD(
            model.layer2.parameters(), 
            lr=0.01
        )
        
        # Train
        print("=======================================================================")
        print("Training layer2...")
        print("=======================================================================")
        self.train_model(model, train_loader_input, val_loader_input, optimizer, epochs=5)
        
        #Freeze all layers again
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze only layer3
        print("Unfreezing layer3...")
        for param in model.layer3.parameters():
            param.requires_grad = True
            
        # Create optimizer for layer3
        optimizer = optim.SGD(
            model.layer3.parameters(), 
            lr=0.04,
        )
        
        # Train
        print("=======================================================================")
        print("Training layer3...")
        print("=======================================================================")
        self.train_model(model, train_loader_input, val_loader_input, optimizer, epochs=5)
        
        #Freeze all layers again
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze only layer4
        print("Unfreezing layer4...")
        for param in model.layer4.parameters():
            param.requires_grad = True
            
        # Create optimizer for layer4
        optimizer = optim.SGD(
            model.layer4.parameters(), 
            lr=0.04,
        )
        
        # Train
        print("=======================================================================")
        print("Training layer4...")
        print("=======================================================================")
        self.train_model(model, train_loader_input, val_loader_input, optimizer, epochs=5)
        
        
        #Freeze all layers again
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze avgpool and fc layers
        print("Unfreezing fc layers...")
        model.fc.requires_grad_(True)
            
        # Create optimizer for fc layer (avgpool has no parameters)
        optimizer = optim.SGD(
            model.fc.parameters(), 
            lr=0.01,
        )

        # Train
        print("=======================================================================")
        print("Training avgpool and fc layers...")
        print("=======================================================================")
        self.train_model(model, train_loader_input, val_loader_input, optimizer, epochs=5)

    def getActivation(self, name):
        def hook(module, input, output):
            self.activation[name] = output.detach()
        return hook
    
    def load_and_use_model_ResNet50(self,path = 'best_model_full_ResNet50.pt', device=torch.device("cpu")):

        #Set up device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #Load the checkpoint
        checkpoint = torch.load(path)

        #Initialize the model architecture
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 2)
        model = model.to(device)

        #Load the saved state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        #Set model to evaluation mode
        model.eval()

        #Transform for new images
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])

        return model, transform
    
    def calculate_cam(self, feature_conv, weight_fc, class_idx):
    
        # generate the class activation maps upsample to 224x224
        size_upsample = (224, 224)
    
        bz, nc, h, w = feature_conv.shape
        print("*****************************")
        print(weight_fc.shape)
        
        print("*****************************")
        

        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam = cv2.resize(cam_img, size_upsample)

        return output_cam

    
    def visualize_cam(self, class_activation_map, width, height, orig_image):

        #A color map that transitions from blue ? cyan ? yellow ? red
        heatmap = cv2.applyColorMap(cv2.resize(class_activation_map,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + orig_image * 0.7
        
        # cv2.imshow("CAM Overlay", result)
        # cv2.waitKey(0)
        # Save the result
        cv2.imwrite("image.jpg", result)


    def extract_image(self, image_path):

        image = cv2.imread(image_path)
        orig_image = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        return image,orig_image, height, width



    def transform_image(self, image):
        transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            ]
        )
        
        # apply the image transforms
        image_tensor = transform(image)
        # add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor

    def predict_image(self, model, image_path):
        

        model.layer4.register_forward_hook(self.getActivation('final_conv'))
        # Load and preprocess the image
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        image,orig_image, height, width = self.extract_image(image_path)
        image_tensor = self.transform_image(image)

        # Make prediction
        with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)

                # Fetch the learned weights at the final feed-forward layer
                conv_features = self.activation['final_conv']
                print("Feature map shape")
                print(conv_features.shape)
                weight_fc = model.fc.weight.detach().numpy()
                print("="*50)
                print(weight_fc.shape)
                
                class_activation_map = self.calculate_cam(conv_features, weight_fc, predicted)
                
                print(f"Model predicted value {predicted.item()}")
                
                
                # visualize result
                self.visualize_cam(class_activation_map, width, height, orig_image)
                
                
    def predict(self, model, test_loader_input):
        test_loader = test_loader_input
        # Load and preprocess the image
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for image, label in test_loader:
            image_tensor = image.to(device)

            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                

                print("-" * 50)
                print(f"Model predicted value {predicted.item()}")
                print(f"Actual value {label.item()}")
                print("-" * 50)
                

    def print_checkpoint_values(self, checkpoint_path='best_model_full_ResNet50.pt'):

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        print("="*50)
        print("CHECKPOINT VALUES")
        print("="*50)
        
        # Print training info
        print(f"Best epoch: {checkpoint['epoch']}")
        print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
        print(f"Validation Accuracy: {checkpoint['val_acc']:.4f}")
        
        
    def collect_model_info(self, model_input, test_loader_input):
        
        model = model_input
        test_loader = test_loader_input

        # 1. Model Architecture
        print("Model Architecture:")
        print(model)
        print("================================================================================")

        # 2. Check Frozen Layers
        print("CHECKING FROZEN AND UNFROZEN LAYERS")
        print("================================================================================")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"Layer {name} is frozen")
            else:
                print(f"Layer {name} is unfrozen")
        print("================================================================================")

        # 3. Dataset Class Information
        print("\nDataset sizes:")
        print(f"Training set: {len(self.train_dataset)} images")
        print(f"Validation set: {len(self.val_dataset)} images")
        print(f"Test set: {len(self.test_dataset)} images")
        print(f"Classes: {self.train_dataset.classes}")
        print(f"Class mapping: {self.train_dataset.class_to_idx}")
        print("================================================================================")

        # 4. Test Loss and Accuracy
        test_loss, test_accuracy = self.test_model(model, test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print("================================================================================")
        
        
    def create_confusion_matrix_plot(self, model_input, test_loader_input, filename="confusion_matrix.png"):
        model = model_input
        test_loader = test_loader_input

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds, labels=np.unique(all_labels))
        cmd = ConfusionMatrixDisplay(cm, display_labels=self.train_dataset.classes)

        # Create a figure and axes with a specific size
        fig, ax = plt.subplots(figsize=(8, 8))  # Adjust figure size as needed
        cmd.plot(ax=ax)

        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title('Confusion Matrix')

        # Adjust layout to make room for labels
        plt.tight_layout()

        plt.savefig(filename)
        return cmd




def main():

    model = ResNet50()
    model.init_dataloader()
    test_loader = model.test_loader
    model_fined_tuned, transform = model.load_and_use_model_ResNet50()
    model.create_confusion_matrix_plot(model_fined_tuned, test_loader)



# def main():

#     model = ResNet50()
#     model.init_dataloader()
#     test_loader = model.test_loader
#     model_fined_tuned, transform = model.load_and_use_model_ResNet50()
#     model.collect_model_info(model_fined_tuned, test_loader)

# def main():
#     model = ResNet50()
#     model.print_checkpoint_values()


# def main():
#     model = ResNet50()
#     model.init_dataloader()
#     model.print_model()


# def main():
#     model = ResNet50()
#     model.init_dataloader()
#     model.print_dataset_class_info()

# def main():

#     model = ResNet50()
#     model.init_dataloader()
#     test_loader = model.test_loader
#     model_fined_tuned, transform = model.load_and_use_model_ResNet50()
#     model.predict(model_fined_tuned, test_loader)


# def main():

#     model = ResNet50()
#     model.init_dataloader()
#     test_loader = model.test_loader
#     model_fined_tuned, transform = model.load_and_use_model_ResNet50()
#     model.predict_image(model_fined_tuned, "NORMAL2-IM-0372-0001.jpeg")




# def main():

#     model = ResNet50()
#     model.init_dataloader()
#     test_loader = model.test_loader
#     model.print_model()
#     model.check_frozen_layer()
#     model.print_dataset_class_info()
#     model.progressive_training_ResNet50(model.model, model.train_loader, model.val_loader)



if __name__ == "__main__":
    main()