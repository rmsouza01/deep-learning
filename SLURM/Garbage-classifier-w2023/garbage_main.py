import torch
import argparse
from garbage_utils import *



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size,  number of images in each iteration during training')
    parser.add_argument('--epochs', type=int, default=100, help='total epochs')
    parser.add_argument('--val_split', type=float, default=0.2, help='val split')
    parser.add_argument('--test_split', type=float, default=0.2, help='test split')
    parser.add_argument('--best_model_path', type=str, help='best model path')
    parser.add_argument('--images_path', type=str, help='path to images')
    parser.add_argument('--verbose', type=bool, default=True, help='verbose debugging flag')
    parser.add_argument('--transfer_learning', type=bool, default=False, help='transfer learning flag')
    
    args = parser.parse_args()

    # Check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    if args.verbose:
      print(device)

    trainloader, valloader, testloader = \
    get_data_loaders(args.images_path, args.val_split, args.test_split,\
                     batch_size=args.batch_size, verbose=args.verbose)


    # Create our model
    net = GarbageModel(4, (3,224,224), args.transfer_learning)
    net.to(device)

    train_validate(net, trainloader, valloader, args.epochs, args.batch_size,\
         args.learning_rate, args.best_model_path, device, args.verbose)

    net.load_state_dict(args.best_model_path)

    test(net, testloader)    

    