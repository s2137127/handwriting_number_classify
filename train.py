import sys
import torch
import torch.nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvNet, MyNet
from data import get_dataloader

if __name__ == "__main__":
    # Specifiy data folder path and model type(fully/conv)
    folder, model_type = sys.argv[1], sys.argv[2]
    print("cuda:　", torch.cuda.is_available())
    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader(folder, batch_size=32)
    # Specify the type of model
    if model_type == 'conv':
        model = ConvNet()
    elif model_type == 'mynet':
        model = MyNet()

    # Set the type of gradient optimizer and the model it update
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.RMSprop(model.parameters(), lr=0.01 ,)
    # Choose loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # use_cuda = torch.cuda.is_available()
    # if use_cuda:
    #     model.cuda()

    # Run any number of epochs you want
    ep = 20
    train_loss = []
    valid_loss = []
    valid_acc = []
    train_acc = []
    best_acc = 0
    cnt = 0
    for epoch in range(ep):
        print('Epoch:', epoch)
        ##############
        ## Training ##
        ##############

        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0


        # Load batch data from dataloader
        for batch, (x, label) in enumerate(train_loader, 1):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            # if use_cuda:
            #     x, label = x.cuda(), label.cuda()
            x, label = x.to(device), label.to(device)
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)
            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch
                print('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))
                train_acc.append(acc)
                train_loss.append(ave_loss)
        ################
        ## Validation ##
        ################
        model.eval()
        # TODO
        loss_val = 0
        total_correct = 0
        total_cnt = 0
        accuracy = 0
        for batch, (x, label) in enumerate(val_loader, 1):
            x, label = x.to(device), label.to(device)
            out = model(x)
            loss = criterion(out, label)
            loss_val += loss.item()
            _, pred_label = torch.max(out, 1)
            total_correct += (pred_label == label).sum().item()
            total_cnt += x.size(0)
            if batch % 150 == 0 or batch == len(val_loader):
                loss_val = loss_val / batch
                accuracy = total_correct / total_cnt
                valid_acc.append(accuracy)
                valid_loss.append(loss_val)
        print("current loss: ", loss_val, "   valid_acc: ", accuracy)
        print("best loss: ", best_acc)
        if accuracy > best_acc:
            cnt = 0
            best_acc = accuracy
            torch.save(model.state_dict(), './checkpoint/%s.pth' % model.name())
        elif best_acc > accuracy:
            cnt += 1
        if cnt > 10:
            print("early stop!!!")
            break


            # Calculate the training loss and accuracy of each iteration


            # Show the training information


        model.train()

    # Save trained model


    # Plot Learning Curve
    # TODO
    fig,ax = plt.subplots(2, 1)
    ax[0].set_title("accuracy")
    # ax[0].legend(loc="upper right")
    ax[0].plot(train_acc,color = 'green')
    ax[0].plot(valid_acc,color = 'red')

    ax[1].set_title("loss")
    # ax[1].legend(loc="upper right")
    ax[1].plot(train_loss,color = 'green')
    ax[1].plot(valid_loss,color = 'red')

    # ax[1,0].set_title("valid_accuracy")
    # ax[1,0].plot(valid_acc)
    # ax[1,1].set_title("valid_loss")
    # ax[1,1].plot(valid_loss)
    plt.show()


