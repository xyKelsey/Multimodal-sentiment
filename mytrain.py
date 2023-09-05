import itertools
import os

import numpy as np, time, random
import datetime
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

from dataloader import get_MELD_loaders
from mymodel_GCN import GCNModel, FocalLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from mymodel_BiRNN import BiRNNModel, RNNModel, MaskedNLLLoss
import warnings
warnings.filterwarnings('ignore')

seed = 6666
learning_rate = 0.001
weight_decay = 1e-5
epoch = 100
batch_size = 16
datafile = 'MELD_features/MELD_features_raw.pkl'
# datafile = 'MELD_features/my_MELD_feature.pkl'

today = datetime.datetime.now()
label_names = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def train_model(model, loss_function, dataloader, model_type='GCN', modality='multimodal', optimizer=None, train=False):
    losses, predicts, labels, masks = [], [], [], []
    loss, output_, label_ = None, None, None
    sample_weight = None
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        text_feature, visual_feature, audio_feature, speaker, label, uttmask = [d for d in data[:-1]]
        utt_num = torch.sum(uttmask, 1).int().tolist() #一个dia里utt的个数 seq_length

        if train:
            optimizer.zero_grad()

        if model_type == 'GCN':
            # utt_num = [uttmask[j].nonzero().tolist()[-1][0] + 1 for j in range(len(uttmask))] #一个dia里utt的个数
            label_ = torch.cat([label[j][:utt_num[j]] for j in range(len(label))])  # 拉成一维，与output的shape匹配
            output_ = model(None, speaker, utt_num, audio_feature, visual_feature, text_feature)


        elif model_type == 'RNN' or model_type == 'BiRNN':
            label_ = label.view(-1)  # batch_size * sequence_length
            if modality == 'bimodal':  # 文本+音频双模态
                feature_concat = torch.cat((text_feature, audio_feature), dim=-1)
            else:  # 文本+音频+视觉多模态
                feature_concat = torch.cat((text_feature, audio_feature, visual_feature), dim=-1)
            output = model(feature_concat, speaker, utt_num)
            output_ = output.transpose(0, 1).contiguous().view(-1, output.size()[2])  # batch_size * sequence_length, num_classes


        loss = loss_function(output_, label_)
        predicts.append(torch.argmax(output_, 1).cpu().numpy())
        labels.append(label_.cpu().numpy())
        losses.append(loss.item())
        masks.append(uttmask.view(-1).numpy())

        if train:
            loss.backward()
            optimizer.step()

    predicts = np.concatenate(predicts)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)

    if model_type == 'GCN':
        sample_weight = None
    else:
        sample_weight = masks

    # ave_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, predicts, sample_weight=sample_weight) * 100, 2)
    avg_fscore = round(f1_score(labels, predicts, sample_weight=sample_weight, average='weighted') * 100, 2)
    class_report = classification_report(labels, predicts, target_names=label_names, sample_weight=sample_weight, digits=4, zero_division=1)
    class_report_with_dict = classification_report(labels, predicts, target_names=label_names, sample_weight=sample_weight, digits=4, output_dict=True, zero_division=1)

    return avg_loss, avg_accuracy, labels, predicts, avg_fscore, masks, class_report, class_report_with_dict

def train_(model_type, modality='multimodal'):
    audio_feature_num = 300
    visual_feature_num = 342
    text_feature_num = 600
    all_feature_num = audio_feature_num + visual_feature_num + text_feature_num
    speakers_num = 9
    classes = 7
    g_hidden = 200

    global_size = 150
    speaker_size = 150
    emotion_size = 100
    r_hidden = 100
    
    class_weights = torch.FloatTensor([4.0, 15.0, 15.0, 3.0, 1.0, 6.0, 3.0])

    train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.1, batch_size=batch_size, datafile=datafile)

    if model_type == 'BiRNN':
        model = BiRNNModel(all_feature_num, global_size, speaker_size, emotion_size, r_hidden, classes)
    elif model_type == 'RNN':
        model = RNNModel(all_feature_num, global_size, speaker_size, emotion_size, r_hidden, classes)
    else:
        model = GCNModel(text_feature_num, visual_feature_num, audio_feature_num, g_hidden,
                        speakers_num=speakers_num, classes=classes, dropout=0.4, alpha=0.2)

    loss_function = FocalLoss()
    # loss_function = MaskedNLLLoss(class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    all_fscore = []

    best_fscore, best_loss, best_label, best_predict, best_mask, test_class_report = None, None, None, None, None, None
    train_losses, train_accuracies, valid_losses, valid_accuracies = [], [], [], []

    for e in range(epoch):
        start_time = time.time()
        train_loss, train_acc, _, _, train_fscore, _, _, _ = train_model(model, loss_function, train_loader,
                                                                         optimizer=optimizer, train=True,
                                                                         model_type=model_type, modality=modality)
        valid_loss, valid_acc, _, _, valid_fscore, _, _, _ = train_model(model, loss_function, valid_loader,
                                                                         model_type=model_type, modality=modality)
        test_loss, test_acc, test_label, test_predict, test_fscore, test_mask, test_class_report, test_class_report_dict = train_model(model, loss_function, test_loader, model_type=model_type, modality=modality)
        all_fscore.append(test_fscore)
        
        # if best_loss == 0 or best_loss > test_loss:
        #     best_loss, best_label, best_predict, best_mask = test_loss, test_label, test_predict, test_mask

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        test_neutral_fscore = test_class_report_dict[label_names[0]]['f1-score']
        test_surprise_fscore = test_class_report_dict[label_names[1]]['f1-score']
        test_fear_fscore = test_class_report_dict[label_names[2]]['f1-score']
        test_sadness_fscore = test_class_report_dict[label_names[3]]['f1-score']
        test_joy_fscore = test_class_report_dict[label_names[4]]['f1-score']
        test_disgust_fscore = test_class_report_dict[label_names[5]]['f1-score']
        test_anger_fscore = test_class_report_dict[label_names[6]]['f1-score']

        if best_loss == None or best_loss > test_loss or \
                (test_neutral_fscore > 0 and test_surprise_fscore > 0 and test_fear_fscore > 0 and test_sadness_fscore \
                and test_joy_fscore > 0 and test_disgust_fscore > 0 and test_anger_fscore > 0):
            best_loss, best_label, best_predict, best_mask, test_class_report = test_loss, test_label, test_predict, test_mask, test_class_report

        outprint = ('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.
                    format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))
        print(outprint)
        # if (e+1)%10 == 0:
        #     print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=1))
        #     print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
        filename = "record_{}_{}_{}_{}.txt".format(model_type, modality, today.day, today.time())
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(outprint)
        else:
            with open(filename, "a") as f:
                f.write("\n" + outprint)

    best_fscore = max(all_fscore)
    
    if model_type == 'GCN':
        sample_weight = None
    else:
        sample_weight = best_mask

    report = classification_report(best_label, best_predict, digits=4, zero_division=1)
    matrix = confusion_matrix(best_label, best_predict, sample_weight=sample_weight)
    with open(filename, "a") as f:
        f.write("\n\nBest_fscore:" + str(best_fscore) + "\nRecord\n" + report)
    print(report)
    print(matrix)
    print('The Best Test F-Score:', best_fscore)
    show_train_history(train_losses, valid_losses, epoch, model_type, modality, loss=True)
    show_train_history(train_accuracies, valid_accuracies, epoch, model_type, modality, loss=False)
    plot_confusion_matrix(matrix, label_names, model_type, modality)

def show_train_history(train, valid, epoch, model_type, modality, loss = False):
    x = [i + 1 for i in range(0, epoch)]
    plt.plot(x, train)
    plt.plot(x, valid)
    plt.title('Train History')
    plt.xlabel('Epoch') 
    if loss:
        plt.ylabel('loss')
    else:
        plt.ylabel('acc(%)')
    plt.legend(['train', 'validation'], loc='upper left')
    if loss:
        plt.savefig('{}_{}_loss.png'.format(model_type, modality))
    else:
        plt.savefig('{}_{}_acc.png'.format(model_type, modality))
    plt.show()
    plt.close()

def plot_confusion_matrix(cm, classes, model_type, modality):
    title = 'Confusion matrix'
    cmap = plt.cm.Blues
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.savefig('{}_{}_confusion_matrix.png'.format(model_type, modality))


if __name__ == '__main__':
    model_type = 'BiRNN' #BiRNN：双向RNN， GCN：图神经网络
    modality = 'multimodal' #multimodal：text+visual+audio， bimodal：text+audio
    train_(model_type, modality)