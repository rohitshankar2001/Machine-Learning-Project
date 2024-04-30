import copy
import numpy as np
from torchvision import transforms
from torch.utils.data import ConcatDataset
from data.manipulate import permutate_image_pixels, SubDataset, TransformedDataset
from data.available import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS
import data.available

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision
from imagecorruptions import corrupt, get_corruption_names
import torch.nn.functional as F
from torchvision.transforms.functional import get_image_size, get_image_num_channels
from PIL import Image
import pickle
import tarfile
import os
import time
from torchvision.datasets import ImageFolder

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir,transform, train = True):
        self.root_dir = root_dir
        self.label_dirs = []
        self.transform = transform
        self.train = train

        if self.train:
            self.data_folder = "train"
        else:
            self.data_folder = "val"

        # self.original_labels = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 
        #           'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01616318', 
        #           'n01622779', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114',
        #           'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01689811', 'n01692333', 'n01693334', 'n01694178', 'n01695060', 
        #           'n01697457', 'n01698640', 'n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381', 'n01740131', 'n01742172', 
        #           'n01748264', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244', 'n01770081', 'n01770393', 'n01773157']
        
        # self.original_labels = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 
        #           'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01616318', 
        #           'n01622779', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114',
        #           'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01689811', 'n01692333', 'n01693334', 'n01694178', 'n01695060', 
        #           'n01748264', 'n01749939']
            
        self.original_labels = ['n04517823', 'n03124043', 'n02117135', 'n01685808', 'n01770393', 'n02100735', 'n04259630', 
                                'n02791270', 'n04270147', 'n02776631', 'n01943899', 'n03980874', 'n03417042', 'n02963159', 'n04162706', 
                                'n01514859', 'n04487394', 'n01843065', 'n02441942', 'n02091244', 'n01873310', 'n02229544', 'n03935335', 
                                'n03393912', 'n02114367', 'n02096437', 'n02111889', 'n04116512', 'n02860847', 'n04525305', 'n07753275', 'n03947888', 
                                'n09472597', 'n02097047', 'n04141076', 'n03854065', 'n03796401', 'n01882714', 'n03481172', 'n01820546', 'n01532829', 
                                'n04380533', 'n01990800', 'n02417914', 'n02676566', 'n02769748', 'n04336792', 'n01950731', 'n04275548', 'n01580077']
        
        #self.original_labels = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475']
        
        self.dirs = []
        for l in self.original_labels:
            self.dirs.append(self.root_dir + "/" + self.data_folder + "/" + l)

        self.images, self.labels = self.load_image_array()

        self.labelsConverted = []
        for i in self.labels:
            index_label = self.original_labels.index(i)
            self.labelsConverted.append(index_label)
        #print(self.labelsConverted)
       
        # self.image_arys = np.array([np.array(Image.open(img_path).convert("RGB")) for img_path in self.images])
        # self.image_arys = [Image.open(img_path).convert("RGB") for img_path in self.images]
        # print("size", len(self.image_arys))
        
    def load_image_array(self):
        image_dirs = []
        label_dirs = []
        for i in self.dirs:
            for root, dirs, files in os.walk(i):
                for file in files:
                    file_path = os.path.join(root, file)
                    image_dirs.append(file_path)
                    label_dirs.append(root[-9:])
        return image_dirs, label_dirs
    
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #print("item getting")
        try:
            image_path = self.images[idx]
            image = Image.open(image_path)
            image = image.convert("RGB")
            
            # image = self.image_arys[idx]

            label = self.labelsConverted[idx]
            image = self.transform(image)

            #label = self.original_labels.index(label)
            return image, label
        except Exception as e:
            print(e)
            idx = idx + 1
            image_path = self.images[idx]
            image = Image.open(image_path)
            image = image.convert("RGB")
            
            # image = self.image_arys[idx]

            label = self.labelsConverted[idx]
            image = self.transform(image)

            #label = self.original_labels.index(label)
            return image, label

    

def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./store/datasets',
                verbose=False, augment=False, normalize=False, target_transform=None, corruption = "",severity = 3,permList= None, seed = 2,arg = None):
    '''Create [train|valid|test]-dataset.'''
    if name == "ImageNet" and type == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = ImageFolder("/mnt/data0/rohit/continual-learning/ImageNet50/train",transform= transform)
        #dataset = ImageDataset("../../../../mnt/ccvl15/ImageNet/", transform = transform, train=True)
        
        for i in range(len(dataset)):
            dataset.targets[i] = permList[dataset.targets[i]]

        if capacity is not None and len(dataset) < capacity:
            dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])

        return dataset
    elif name == "ImageNet" and type == 'test':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        #testset = ImageDataset("../../../../mnt/ccvl15/ImageNet/", transform = transform, train=False)
        testset = ImageFolder("/mnt/data0/rohit/continual-learning/ImageNet50/test",transform= transform)

        for i in range(len(testset)):
            testset.targets[i] = permList[testset.targets[i]]

        if capacity is not None and len(testset) < capacity:
            testset = ConcatDataset([copy.deepcopy(testset) for _ in range(int(np.ceil(capacity / len(testset))))])


        corrupted_dataset_list = []
        # corruption_list = get_corruption_names()
        # print(corruption_list)
        # for sev in range(5):
        #     for index, cor in enumerate(corruption_list):
        #         # if index == 15:
        #         #     break
        #         corruption_transform = None
        #         print(cor)
                
        #         if name == "ImageNet":
        #             corruption_transform = transforms.Compose([
        #                 #data.available.corruption(cor, sev + 1),
        #                 transforms.ToTensor(),
        #                 transforms.Resize((224, 224)),
        #                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #             ])   

        #         print(name)
        #         test_dataset2 = ImageDataset("../../../../mnt/ccvl15/ImageNet/", transform = corruption_transform, train=False)

        #         for i in range(len(test_dataset2)):
        #             test_dataset2.labelsConverted[i] = permList[test_dataset2.labelsConverted[i]]

        #         if verbose:
        #             print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(test_dataset2)))

        #         if capacity is not None and len(test_dataset2) < capacity:
        #             test_dataset2 = ConcatDataset([copy.deepcopy(test_dataset2) for _ in range(int(np.ceil(capacity / len(test_dataset2))))])

        #         corrupted_dataset_list.append(test_dataset2)

        return testset, corrupted_dataset_list


    else:
        data_name = 'MNIST' if name in ('MNIST28', 'MNIST32') else name
        dataset_class = AVAILABLE_DATASETS[data_name]

        # specify image-transformations to be applied
        transforms_list = [*AVAILABLE_TRANSFORMS['augment']] if augment else []
        transforms_list += [*AVAILABLE_TRANSFORMS[name]]

        if normalize:
            transforms_list += [*AVAILABLE_TRANSFORMS[name+"_norm"]]
        if permutation is not None:
            transforms_list.append(transforms.Lambda(lambda x, p=permutation: permutate_image_pixels(x, p)))
        dataset_transform = transforms.Compose(transforms_list)
    


    # Corrupting test set here
    if type == 'test'and (name == "CIFAR100" or name == "CIFAR10" or name == "x"): # and corruption != "" : 
        # try:
        #     load_set = torch.load(corruption + arg.scenario + str(severity) + name + str(seed)+ str(arg.pre_convE) + '.pth')
        #     print("corrupted set loaded")
        #     return load_set
        # except FileNotFoundError:
        print("DATASET FOR TEST")


        test_dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name),train = False,
                            download=download, transform=dataset_transform, target_transform=None)
        print(name)
        if name != "MNIST32":
            for i in range(len(test_dataset)):
                test_dataset.targets[i] = permList[test_dataset.targets[i]]

        if verbose:
            print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(test_dataset)))

        if capacity is not None and len(test_dataset) < capacity:
            test_dataset = ConcatDataset([copy.deepcopy(test_dataset) for _ in range(int(np.ceil(capacity / len(test_dataset))))])

        corrupted_dataset_list = []
        corruption_list = get_corruption_names()
        print(corruption_list)
        for sev in range(5):
            for index, cor in enumerate(corruption_list):
                # if index == 15:
                #     break
                corruption_transform = None
                print(cor)
                if name == "CIFAR100":
                    corruption_transform = transforms.Compose([
                        data.available.corruption(cor, sev + 1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
                    ])
                if name == "CIFAR10":
                    corruption_transform = transforms.Compose([
                        data.available.corruption(cor, sev + 1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
                    ])    
                
             

                print(name)
                test_dataset2 = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name),train = False,
                                    download=download, transform=corruption_transform, target_transform=None)
                #image, label = test_dataset2[0]



                if name != "MNIST32":
                    for i in range(len(test_dataset2)):
                        test_dataset2.targets[i] = permList[test_dataset2.targets[i]]

                if verbose:
                    print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(test_dataset2)))

                if capacity is not None and len(test_dataset2) < capacity:
                    test_dataset2 = ConcatDataset([copy.deepcopy(test_dataset2) for _ in range(int(np.ceil(capacity / len(test_dataset2))))])

                corrupted_dataset_list.append(test_dataset2)

        return test_dataset, corrupted_dataset_list



    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=None)
    
    #print(dataset.data[0])
    # plt.title('data Image')
    # plt.imshow(dataset.data[0])
    # plt.show()
    # print(name)
    
    if name != "MNIST32":
        for i in range(len(dataset)):
            print("TEST")
            print(permList)
            print(dataset.targets)
            dataset.targets[i] = permList[dataset.targets[i]]
        
    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])


    
    return dataset

#----------------------------------------------------------------------------------------------------------#

def get_singlecontext_datasets(name, data_dir="./store/datasets", normalize=False, augment=False, verbose=False):
    '''Load, organize and return train- and test-dataset for requested single-context experiment.'''

    # Get config-dict and data-sets
    config = DATASET_CONFIGS[name]
    config['output_units'] = config['classes']
    config['normalize'] = normalize
    if normalize:
        config['denormalize'] = AVAILABLE_TRANSFORMS[name+"_denorm"]
    trainset = get_dataset(name, type='train', dir=data_dir, verbose=verbose, normalize=normalize, augment=augment)
    testset = get_dataset(name, type='test', dir=data_dir, verbose=verbose, normalize=normalize)

    # Return tuple of data-sets and config-dictionary
    return (trainset, testset), config

#----------------------------------------------------------------------------------------------------------#



def get_context_set(name, scenario, contexts, data_dir="./datasets", only_config=False, verbose=False,
                    exception=False, normalize=False, augment=False, singlehead=False, train_set_per_class=False, corruption="",severity = 3, seed = 2, arg = None):
    '''Load, organize and return a context set (both train- and test-data) for the requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first context (permMNIST) or digits
                            are not shuffled before being distributed over the contexts (e.g., splitMNIST, CIFAR100)'''

    ## NOTE: options 'normalize' and 'augment' only implemented for CIFAR-based experiments.

    # Define data-type
    if name == "splitMNIST":
        data_type = 'MNIST'
    elif name == "permMNIST":
        data_type = 'MNIST32'
        if train_set_per_class:
            raise NotImplementedError('Permuted MNIST currently has no support for separate training dataset per class')
    elif name == "CIFAR10":
        data_type = 'CIFAR10'
    elif name == "CIFAR100":
        data_type = 'CIFAR100'
    elif name == "ImageNet":
        data_type = "ImageNet"
    else:
        raise ValueError('Given undefined experiment: {}'.format(name))

    # Get config-dict
    config = DATASET_CONFIGS[data_type].copy()
    config['normalize'] = normalize if name=='CIFAR100' else False
    if config['normalize']:
        config['denormalize'] = AVAILABLE_TRANSFORMS["CIFAR100_denorm"]
    # check for number of contexts
    if contexts > config['classes'] and not name=="permMNIST":
        raise ValueError("Experiment '{}' cannot have more than {} contexts!".format(name, config['classes']))
    # -how many classes per context?
    classes_per_context = 10 if name=="permMNIST" else int(np.floor(config['classes'] / contexts))
    config['classes_per_context'] = classes_per_context
    config['output_units'] = classes_per_context if (scenario=='domain' or
                                                    (scenario=="task" and singlehead)) else classes_per_context*contexts
    # -if only config-dict is needed, return it
    if only_config:
        return config
    # Depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # get train and test datasets
        trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=None, verbose=verbose)
        testset, testset2 = get_dataset(data_type, type="test", dir=data_dir, target_transform=None, verbose=True, corruption=corruption)
        
        # generate pixel-permutations
        if exception:
            permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(contexts-1)]
        else:
            permutations = [np.random.permutation(config['size']**2) for _ in range(contexts)]
        # specify transformed datasets per context
        train_datasets = []
        test_datasets  = []
        test_datasets2  = []
        test_datasets3  = []
        test_datasets4  = []
        test_datasets5  = []
        test_datasets6  = []
        test_datasets7  = []
        test_datasets8  = []
        test_datasets9  = []
        test_datasets10 = []
        test_datasets11 = []
        test_datasets12 = []
        test_datasets13 = []
        test_datasets14 = []
        test_datasets15 = []
        test_datasets16 = []
        test_datasets17 = []
        test_datasets18 = []
        test_datasets19 = []
        test_datasets20 = []
        test_datasets21 = []
        test_datasets22 = []
        test_datasets23 = []
        test_datasets24 = []
        test_datasets25 = []
        test_datasets26 = []
        test_datasets27 = []
        test_datasets28 = []
        test_datasets29 = []
        test_datasets30 = []
        test_datasets31 = []
        test_datasets32 = []
        test_datasets33 = []
        test_datasets34 = []
        test_datasets35 = []
        test_datasets36 = []
        test_datasets37 = []
        test_datasets38 = []
        test_datasets39 = []
        test_datasets40 = []
        test_datasets41 = []
        test_datasets42 = []
        test_datasets43 = []
        test_datasets44 = []
        test_datasets45 = []
        test_datasets46 = []
        test_datasets47 = []
        test_datasets48 = []
        test_datasets49 = []
        test_datasets50 = []
        test_datasets51 = []
        test_datasets52 = []
        test_datasets53 = []
        test_datasets54 = []
        test_datasets55 = []
        test_datasets56 = []
        test_datasets57 = []
        test_datasets58 = []
        test_datasets59 = []
        test_datasets60 = []
        test_datasets61 = []
        test_datasets62 = []
        test_datasets63 = []
        test_datasets64 = []
        test_datasets65 = []
        test_datasets66 = []
        test_datasets67 = []
        test_datasets68 = []
        test_datasets69 = []
        test_datasets70 = []
        test_datasets71 = []
        test_datasets72 = []
        test_datasets73 = []
        test_datasets74 = []
        test_datasets75 = []
        test_datasets76 = []


        for context_id, perm in enumerate(permutations):

            # target_transform = transforms.Lambda(
            #     lambda y, x=context_id: y + x*classes_per_context
            # ) if scenario in ('task', 'class') and not (scenario=='task' and singlehead) else None
            
            target_transform = None
            if scenario in ('task', 'class') and not (scenario=='task' and singlehead):
                target_transform = data.available.multContextClass(context_id,classes_per_context)


            # train_datasets.append(TransformedDataset(
            #     trainset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
            #     target_transform=target_transform
            # ))
                
            train_datasets.append(TransformedDataset(
                trainset, transform=data.available.permute_img(perm),
                target_transform=target_transform
            ))

            # test_datasets.append(TransformedDataset(
            #     testset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
            #     target_transform=target_transform
            # ))
            print(context_id)
            #test_datasets.append(TransformedDataset(  testset     , transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets2.append(TransformedDataset( testset2[0] , transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets3.append(TransformedDataset( testset2[1] , transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets4.append(TransformedDataset( testset2[2] , transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets5.append(TransformedDataset( testset2[3] , transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets6.append(TransformedDataset( testset2[4] , transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets7.append(TransformedDataset( testset2[5] , transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets8.append(TransformedDataset( testset2[6] , transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets9.append(TransformedDataset( testset2[7] , transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets10.append(TransformedDataset( testset2[8] , transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets11.append(TransformedDataset( testset2[9] , transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets12.append(TransformedDataset( testset2[10], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets13.append(TransformedDataset( testset2[11], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets14.append(TransformedDataset( testset2[12], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets15.append(TransformedDataset( testset2[13], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets16.append(TransformedDataset( testset2[14], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets17.append(TransformedDataset( testset2[15], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets18.append(TransformedDataset( testset2[16], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets19.append(TransformedDataset( testset2[17], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets20.append(TransformedDataset( testset2[18], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets21.append(TransformedDataset( testset2[19], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets22.append(TransformedDataset( testset2[20], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets23.append(TransformedDataset( testset2[21], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets24.append(TransformedDataset( testset2[22], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets25.append(TransformedDataset( testset2[23], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets26.append(TransformedDataset( testset2[24], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets27.append(TransformedDataset( testset2[25], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets28.append(TransformedDataset( testset2[26], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets29.append(TransformedDataset( testset2[27], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets30.append(TransformedDataset( testset2[28], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets31.append(TransformedDataset( testset2[29], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets32.append(TransformedDataset( testset2[30], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets33.append(TransformedDataset( testset2[31], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets34.append(TransformedDataset( testset2[32], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets35.append(TransformedDataset( testset2[33], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets36.append(TransformedDataset( testset2[34], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets37.append(TransformedDataset( testset2[35], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets38.append(TransformedDataset( testset2[36], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets39.append(TransformedDataset( testset2[37], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets40.append(TransformedDataset( testset2[38], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets41.append(TransformedDataset( testset2[39], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets42.append(TransformedDataset( testset2[40], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets43.append(TransformedDataset( testset2[41], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets44.append(TransformedDataset( testset2[42], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets45.append(TransformedDataset( testset2[43], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets46.append(TransformedDataset( testset2[44], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets47.append(TransformedDataset( testset2[45], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets48.append(TransformedDataset( testset2[46], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets49.append(TransformedDataset( testset2[47], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets50.append(TransformedDataset( testset2[48], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets51.append(TransformedDataset( testset2[49], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets52.append(TransformedDataset( testset2[50], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets53.append(TransformedDataset( testset2[51], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets54.append(TransformedDataset( testset2[52], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets55.append(TransformedDataset( testset2[53], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets56.append(TransformedDataset( testset2[54], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets57.append(TransformedDataset( testset2[55], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets58.append(TransformedDataset( testset2[56], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets59.append(TransformedDataset( testset2[57], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets60.append(TransformedDataset( testset2[58], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets61.append(TransformedDataset( testset2[59], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets62.append(TransformedDataset( testset2[60], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets63.append(TransformedDataset( testset2[61], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets64.append(TransformedDataset( testset2[62], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets65.append(TransformedDataset( testset2[63], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets66.append(TransformedDataset( testset2[64], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets67.append(TransformedDataset( testset2[65], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets68.append(TransformedDataset( testset2[66], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets69.append(TransformedDataset( testset2[67], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets70.append(TransformedDataset( testset2[68], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets71.append(TransformedDataset( testset2[69], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets72.append(TransformedDataset( testset2[70], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets73.append(TransformedDataset( testset2[71], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets74.append(TransformedDataset( testset2[72], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets75.append(TransformedDataset( testset2[73], transform=data.available.permute_img(perm),target_transform=target_transform))
            test_datasets76.append(TransformedDataset( testset2[74], transform=data.available.permute_img(perm),target_transform=target_transform))


        return ((train_datasets, test_datasets, 
        #     test_datasets2,
        #     test_datasets3,
        #     test_datasets4,
        #     test_datasets5,
        #     test_datasets6,
        #     test_datasets7,
        #     test_datasets8,
        #     test_datasets9,
        #     test_datasets10, 
        #     test_datasets11, 
        #     test_datasets12,
        #     test_datasets13,
        #     test_datasets14,
        #     test_datasets15,
        #     test_datasets16,
        # test_datasets17,
        # test_datasets18,
        # test_datasets19,
        # test_datasets20,
        # test_datasets21,
        # test_datasets22,
        # test_datasets23,
        # test_datasets24,
        # test_datasets25,
        # test_datasets26,
        # test_datasets27,
        # test_datasets28,
        # test_datasets29,
        # test_datasets30,
        # test_datasets31,
        # test_datasets32,
        # test_datasets33,
        # test_datasets34,
        # test_datasets35,
        # test_datasets36,
        # test_datasets37,
        # test_datasets38,
        # test_datasets39,
        # test_datasets40,
        # test_datasets41,
        # test_datasets42,
        # test_datasets43,
        # test_datasets44,
        # test_datasets45,
        # test_datasets46,
        # test_datasets47,
        # test_datasets48,
        # test_datasets49,
        # test_datasets50,
        # test_datasets51,
        # test_datasets52,
        # test_datasets53,
        # test_datasets54,
        # test_datasets55,
        # test_datasets56,
        # test_datasets57,
        # test_datasets58,
        # test_datasets59,
        # test_datasets60,
        # test_datasets61,
        # test_datasets62,
        # test_datasets63,
        # test_datasets64,
        # test_datasets65,
        # test_datasets66,
        # test_datasets67,
        # test_datasets68,
        # test_datasets69,
        # test_datasets70,
        # test_datasets71,
        # test_datasets72,
        # test_datasets73,
        # test_datasets74,
        # test_datasets75,
        # test_datasets76

            ), config)

    else:
        # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
        classes = config['classes']
        perm_class_list = np.array(list(range(classes))) if exception else np.random.permutation(list(range(classes)))
        target_transform = transforms.Lambda(lambda y, p=perm_class_list: int(p[y]))
        # prepare train and test datasets with all classes
        try:
            print("speed load")
            print("trying")

            if arg.scenario == "domain":
                print("Working")
                train_datasets =   torch.load("loadable_sets/"+ "Domain" + "train" + name + str(2) +'.pth')
                test_datasets  =   torch.load("loadable_sets/"+ "Domain" + "test" + name + str(2)  +'.pth')
                #test_datasets2  =   torch.load("loadable_sets/"+ "Domain" + "test123123" + name + str(2)  +'.pth')

                print("speed load completed")

            else: 
                print("loading set trying ")
                train_datasets = torch.load("loadable_sets/" + "train" + name + str(2) +'.pth')
                test_datasets  = torch.load("loadable_sets/" + "test" + name + str(2)  +'.pth')

                # test_datasets = torch.load("loadable_sets/" + "x" + name + str(22)  +'.pth')


                print("speed load completed")

        except Exception as e:
            print(e)
            # print("new sets creating")
            # print(perm_class_list)
            print("new_set_creation")
            
            trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=target_transform,
                                verbose=verbose, augment=augment, normalize=normalize, permList= perm_class_list)
            testset, testset2 = get_dataset(data_type, type="test", dir=data_dir, target_transform=target_transform, verbose=verbose,
                                augment=augment, normalize=normalize, corruption=corruption, severity = severity, permList= perm_class_list, seed= seed, arg=arg)


            # generate labels-per-dataset (if requested, training data is split up per class rather than per context)
            print("train_set_per_class:", train_set_per_class)
            labels_per_dataset_train = [[label] for label in range(classes)] if train_set_per_class else [
                list(np.array(range(classes_per_context))+classes_per_context*context_id) for context_id in range(contexts)
            ]
            labels_per_dataset_test = [
                list(np.array(range(classes_per_context))+classes_per_context*context_id) for context_id in range(contexts)
            ]
            # split the train and test datasets up into sub-datasets
            
            test_datasets = []
            test_datasets2 = []
            train_datasets = []

            print("train")
            print(labels_per_dataset_train)
            for labels in labels_per_dataset_train:
                print(labels)
                target_trans = None
                if  scenario=='domain' or (scenario=='task' and singlehead):
                    target_trans = data.available.subtractLabel(labels[0])
                train_datasets.append(SubDataset(trainset, labels, target_transform=target_trans))
            if arg.scenario == "domain":
                torch.save(train_datasets,"loadable_sets/"+"Domain" + "train" + name + str(2) +'.pth')
            else:
                torch.save(train_datasets,"loadable_sets/" + "train" + name + str(2) +'.pth')
            

            print("test")
            #print(labels_per_dataset_test)
            for labels in labels_per_dataset_test:
                # target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if (
                #         scenario=='domain' or (scenario=='task' and singlehead)
                # ) else None
                # print("labels")
                # print(labels)
                target_trans = None
                if  scenario=='domain' or (scenario=='task' and singlehead):
                    target_trans = data.available.subtractLabel(labels[0])

                test_datasets.append(SubDataset(testset, labels, target_transform=target_trans))
                print(labels)

            if arg.scenario == "domain":
                torch.save(test_datasets,   "loadable_sets/"+ "Domain" +  "test" + name + str(seed)  +'.pth')

    # Return tuple of train- and test-dataset, config-dictionary and number of classes per context
    return ((train_datasets, test_datasets, 
            # test_datasets2,
            # test_datasets3,
            # test_datasets4,
            # test_datasets5,
            # test_datasets6,
            # test_datasets7,
            # test_datasets8,
            # test_datasets9,
            # test_datasets10, 
            # test_datasets11, 
            # test_datasets12,
            # test_datasets13,
            # test_datasets14,
            # test_datasets15,
            # test_datasets16,
            # test_datasets17,
            # test_datasets18,
            # test_datasets19,
            # test_datasets20,
            # test_datasets21,
            # test_datasets22,
            # test_datasets23,
            # test_datasets24,
            # test_datasets25,
            # test_datasets26,
            # test_datasets27,
            # test_datasets28,
            # test_datasets29,
            # test_datasets30,
            # test_datasets31,
            # test_datasets32,
            # test_datasets33,
            # test_datasets34,
            # test_datasets35,
            # test_datasets36,
            # test_datasets37,
            # test_datasets38,
            # test_datasets39,
            # test_datasets40,
            # test_datasets41,
            # test_datasets42,
            # test_datasets43,
            # test_datasets44,
            # test_datasets45,
            # test_datasets46,
            # test_datasets47,
            # test_datasets48,
            # test_datasets49,
            # test_datasets50,
            # test_datasets51,
            # test_datasets52,
            # test_datasets53,
            # test_datasets54,
            # test_datasets55,
            # test_datasets56,
            # test_datasets57,
            # test_datasets58,
            # test_datasets59,
            # test_datasets60,
            # test_datasets61,
            # test_datasets62,
            # test_datasets63,
            # test_datasets64,
            # test_datasets65,
            # test_datasets66,
            # test_datasets67,
            # test_datasets68,
            # test_datasets69,
            # test_datasets70,
            # test_datasets71,
            # test_datasets72,
            # test_datasets73,
            # test_datasets74,
            # test_datasets75,
            # test_datasets76
             ), config)