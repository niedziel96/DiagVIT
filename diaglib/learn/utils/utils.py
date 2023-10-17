import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler

def get_optim(model, args):
	if args.optimizer == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
	elif args.optimizer == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	else:
		raise NotImplementedError
	return optimizer
	
def collate_fct(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]
	
def get_split_loader(split_dataset, args, training = False, weighted = False):
	"""
		return either the validation loader or training loader 
	"""

	kwargs = {'num_workers': 8} if torch.device.type == "cuda" else {}

	if training:
		if weighted:
			weights = make_weights_for_balanced_classes_split(split_dataset)
			loader = DataLoader(split_dataset, batch_size=args.batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_fct, **kwargs)	
		else:
			loader = DataLoader(split_dataset, batch_size=args.batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate_fct, **kwargs)
	else:
		loader = DataLoader(split_dataset, batch_size=args.batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_fct, **kwargs)


	return loader
	
def make_weights_for_balanced_classes_split(dataset):
    N = int(float(len(dataset)))  
    print(f'----- there is in total {N} datapoints -----')  
    cl_ratios = dataset.get_class_ratios()
    print(f'----- calculated ratios: {cl_ratios} -----')
    weight_per_class = [value for key, value in cl_ratios.items()]    
    #weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
	
    
    for idx in range(N):   
        y = dataset.input_data_table['label'][idx]    
        weight[idx] = weight_per_class[int(y)-1]     
                            

    return torch.DoubleTensor(weight)

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)
 
def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error