# cell 1
import torch
import matplotlib.pyplot as plt
# Import your class from the python file
from resnet_model import ResNet50

resnet = ResNet50()
resnet.init_dataloader()

model, transform = resnet.load_and_use_model_ResNet50(path='best_model_full_ResNet50.pt')
test_loss, test_accuracy = resnet.test_model(model, resnet.test_loader)
print(f"test_loss {test_loss}")
print(f"test_accuracy {test_accuracy}\n\n")

eval_loss, eval_accuracy = resnet.eval_model(model, resnet.val_loader)
print(f"eval_loss {eval_loss}")
print(f"eval_accuracy {eval_accuracy}\n\n")

all_labels, all_preds, all_probs = resnet.evaluator(model, resnet.test_loader)
precisions, recalls, F1s = resnet.compute_metrics(all_labels, all_preds)
        
print(f"precisionsNormal {precisions[0]}")
print(f"precisionsPneumonia {precisions[1]}\n\n")
        
print(f"RecallNormal {recalls[0]}")
print(f"RecallPneumonia {recalls[1]}\n\n")
        
print(f"F1ScoreNormal {F1s[0]}")
print(f"F1ScorePneumonia {F1s[1]}\n\n")
        
resnet.plot_roc(all_labels, all_probs)

resnet.create_confusion_matrix_plot(all_labels, all_preds)

resnet.print_dataset_class_info()