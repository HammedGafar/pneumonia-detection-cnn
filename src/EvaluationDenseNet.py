# cell 1
import torch
import matplotlib.pyplot as plt
# Import your class from the python file
from DenseNet_model import DenseNet161

DenseNet = DenseNet161()
DenseNet.init_dataloader()
model, transform = DenseNet.load_and_use_model_DenseNet(path='best_model_full_DenseNet.pt')
test_loss, test_accuracy = DenseNet.test_model(model, DenseNet.test_loader)
print(f"test_loss {test_loss}")
print(f"test_accuracy {test_accuracy}\n\n")

eval_loss, eval_accuracy = DenseNet.eval_model(model, DenseNet.val_loader)
print(f"eval_loss {eval_loss}")
print(f"eval_accuracy {eval_accuracy}\n\n")

all_labels, all_preds, all_probs = DenseNet.evaluator(model, DenseNet.test_loader)
precisions, recalls, F1s = DenseNet.compute_metrics(all_labels, all_preds)
        
print(f"precisionsNormal {precisions[0]}")
print(f"precisionsPneumonia {precisions[1]}\n\n")
        
print(f"RecallNormal {recalls[0]}")
print(f"RecallPneumonia {recalls[1]}\n\n")
        
print(f"F1ScoreNormal {F1s[0]}")
print(f"F1ScorePneumonia {F1s[1]}\n\n")
        
DenseNet.create_confusion_matrix_plot(all_labels, all_preds)