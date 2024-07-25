import random
import numpy as np
import torch


class SingletonMeta(type):
    _instances = {}  # Dictionary to hold the single instances of each class using this metaclass.

    def __call__(cls, *args, **kwargs):
        # __call__ is invoked when an instance of a class is called.
        # cls here refers to the class that is being instantiated.

        if cls not in cls._instances:
            # If the class is not already in the _instances dictionary,
            # call the parent __call__ method to create an instance.
            instance = super().__call__(*args, **kwargs)
            
            # Store the created instance in the _instances dictionary.
            cls._instances[cls] = instance
            
        # Return the stored instance.
        return cls._instances[cls]
    
    
class PyRandomWrapper(metaclass=SingletonMeta):
    seed_value = None
    random_counter = 0
    operation_counter = {
        'randint': 0,
        'choice': 0,
        'choices': 0,
        'shuffle': 0,
        'random': 0, 
        'uniform': 0
    }
    
    @staticmethod
    def seed(seed):
        PyRandomWrapper.seed_value = seed
        random.seed(seed)
    
    @staticmethod
    def randint(a, b):
        PyRandomWrapper.random_counter += 1
        PyRandomWrapper.operation_counter['randint'] += 1
        return random.randint(a, b)
    
    @staticmethod
    def choice(seq):
        PyRandomWrapper.random_counter += 1
        PyRandomWrapper.operation_counter['choice'] += 1
        return random.choice(seq)
    
    @staticmethod
    def choices(seq, k):
        PyRandomWrapper.random_counter += 1
        PyRandomWrapper.operation_counter['choices'] += 1
        return random.choices(seq, k=k)
    
    @staticmethod
    def shuffle(seq):
        PyRandomWrapper.random_counter += 1
        PyRandomWrapper.operation_counter['shuffle'] += 1
        random.shuffle(seq)
    
    @staticmethod
    def random():
        PyRandomWrapper.random_counter += 1
        PyRandomWrapper.operation_counter['random'] += 1
        return random.random()
    
    @staticmethod
    def uniform(a, b):
        PyRandomWrapper.random_counter += 1
        PyRandomWrapper.operation_counter['uniform'] += 1
        return random.uniform(a, b)
    
    @staticmethod
    def get_seed():
        return PyRandomWrapper.seed_value
    
    @staticmethod
    def get_random_counter():
        return PyRandomWrapper.random_counter
    
    @staticmethod
    def get_operation_counter():
        return PyRandomWrapper.operation_counter
    
    @staticmethod
    def reset_random_counter():
        PyRandomWrapper.random_counter = 0
        for key in PyRandomWrapper.operation_counter:
            PyRandomWrapper.operation_counter[key] = 0



class NumpyRandomWrapper(metaclass=SingletonMeta):
    seed_value = None
    random_counter = 0
    operation_counter = {
        'rand': 0,
        'randint': 0,
        'choice': 0,
        'shuffle': 0,
        'random': 0,
        'normal': 0,
        'uniform': 0, 
        'beta': 0,
        'permutation': 0
    }
    
    @staticmethod
    def seed(seed):
        NumpyRandomWrapper.seed_value = seed
        np.random.seed(seed)
    
    @staticmethod
    def randint(low, high=None, size=None, dtype=int):
        NumpyRandomWrapper.random_counter += 1
        NumpyRandomWrapper.operation_counter['randint'] += 1
        return np.random.randint(low, high=high, size=size, dtype=dtype)
    
    def rand(d0, d1=None, d2=None):
        NumpyRandomWrapper.random_counter += 1
        NumpyRandomWrapper.operation_counter['rand'] += 1
        return np.random.rand(d0, d1, d2)
    
    @staticmethod
    def choice(a, size=None, replace=True, p=None):
        NumpyRandomWrapper.random_counter += 1
        NumpyRandomWrapper.operation_counter['choice'] += 1
        return np.random.choice(a, size=size, replace=replace, p=p)
    
    @staticmethod
    def shuffle(x):
        NumpyRandomWrapper.random_counter += 1
        NumpyRandomWrapper.operation_counter['shuffle'] += 1
        np.random.shuffle(x)
    
    @staticmethod
    def random(size=None):
        NumpyRandomWrapper.random_counter += 1
        NumpyRandomWrapper.operation_counter['random'] += 1
        return np.random.random(size)
    
    @staticmethod
    def normal(loc=0.0, scale=1.0, size=None):
        NumpyRandomWrapper.random_counter += 1
        NumpyRandomWrapper.operation_counter['normal'] += 1
        return np.random.normal(loc, scale, size)
    
    @staticmethod
    def uniform(low=0.0, high=1.0, size=None):
        NumpyRandomWrapper.random_counter += 1
        NumpyRandomWrapper.operation_counter['uniform'] += 1
        return np.random.uniform(low, high, size)
    
    @staticmethod
    def beta(a, b, size=None):
        NumpyRandomWrapper.random_counter += 1
        NumpyRandomWrapper.operation_counter['beta'] += 1
        return np.random.beta(a, b, size)
    
    @staticmethod
    def permutation(x):
        NumpyRandomWrapper.random_counter += 1
        NumpyRandomWrapper.operation_counter['permutation'] += 1
        return np.random.permutation(x)
    
    @staticmethod
    def get_seed():
        return NumpyRandomWrapper.seed_value
    
    @staticmethod
    def get_random_counter():
        return NumpyRandomWrapper.random_counter
    
    @staticmethod
    def get_operation_counter():
        return NumpyRandomWrapper.operation_counter
    
    @staticmethod
    def reset_random_counter():
        NumpyRandomWrapper.random_counter = 0
        for key in NumpyRandomWrapper.operation_counter:
            NumpyRandomWrapper.operation_counter[key] = 0
       
       
            
class TorchRandomWrapper(metaclass=SingletonMeta):
    seed_value = None
    random_counter = 0
    operation_counter = {
        'manual_seed': 0,
        'cuda_manual_seed_all': 0,
        'rand': 0,
        'randn': 0,
        'randint': 0,
        'multinomial': 0,
        'bernoulli': 0,
        'randperm': 0,
        'rand_like': 0,
        'randn_like': 0,
        'randint_like': 0
    }
    
    @staticmethod
    def seed(seed):
        TorchRandomWrapper.seed_value = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        TorchRandomWrapper.random_counter += 1
        TorchRandomWrapper.operation_counter['manual_seed'] += 1
        TorchRandomWrapper.operation_counter['cuda_manual_seed_all'] += 1
    
    @staticmethod
    def manual_seed(seed):
        torch.manual_seed(seed)
        TorchRandomWrapper.random_counter += 1
        TorchRandomWrapper.operation_counter['manual_seed'] += 1
    
    @staticmethod
    def cuda_manual_seed_all(seed):
        torch.cuda.manual_seed_all(seed)
        TorchRandomWrapper.random_counter += 1
        TorchRandomWrapper.operation_counter['cuda_manual_seed_all'] += 1
    
    @staticmethod
    def rand(*args, **kwargs):
        TorchRandomWrapper.random_counter += 1
        TorchRandomWrapper.operation_counter['rand'] += 1
        return torch.rand(*args, **kwargs)
    
    @staticmethod
    def randn(*args, **kwargs):
        TorchRandomWrapper.random_counter += 1
        TorchRandomWrapper.operation_counter['randn'] += 1
        return torch.randn(*args, **kwargs)
    
    @staticmethod
    def randint(low, high=None, size=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False):
        TorchRandomWrapper.random_counter += 1
        TorchRandomWrapper.operation_counter['randint'] += 1
        return torch.randint(low, high, size, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    
    @staticmethod
    def multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
        TorchRandomWrapper.random_counter += 1
        TorchRandomWrapper.operation_counter['multinomial'] += 1
        return torch.multinomial(input, num_samples, replacement, generator=generator, out=out)
    
    @staticmethod
    def bernoulli(input, *, generator=None, out=None):
        TorchRandomWrapper.random_counter += 1
        TorchRandomWrapper.operation_counter['bernoulli'] += 1
        return torch.bernoulli(input, generator=generator, out=out)
    
    @staticmethod
    def randperm(n, *, generator=None, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False):
        TorchRandomWrapper.random_counter += 1
        TorchRandomWrapper.operation_counter['randperm'] += 1
        return torch.randperm(n, generator=generator, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    
    @staticmethod
    def rand_like(input, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format):
        TorchRandomWrapper.random_counter += 1
        TorchRandomWrapper.operation_counter['rand_like'] += 1
        return torch.rand_like(input, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)
    
    @staticmethod
    def randn_like(input, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format):
        TorchRandomWrapper.random_counter += 1
        TorchRandomWrapper.operation_counter['randn_like'] += 1
        return torch.randn_like(input, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)
    
    @staticmethod
    def randint_like(input, low=0, high=None, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format):
        TorchRandomWrapper.random_counter += 1
        TorchRandomWrapper.operation_counter['randint_like'] += 1
        return torch.randint_like(input, low, high, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)
    
    @staticmethod
    def get_seed():
        return TorchRandomWrapper.seed_value
    
    @staticmethod
    def get_random_counter():
        return TorchRandomWrapper.random_counter
    
    @staticmethod
    def get_operation_counter():
        return TorchRandomWrapper.operation_counter
    
    @staticmethod
    def reset_random_counter():
        TorchRandomWrapper.random_counter = 0
        for key in TorchRandomWrapper.operation_counter:
            TorchRandomWrapper.operation_counter[key] = 0


# Usage Example
if __name__ == '__main__':
    PyRandomWrapper.seed(42)
    print(PyRandomWrapper.get_seed())                 # 42
    
    print(PyRandomWrapper.randint(1, 10))              
    print(PyRandomWrapper.choice([1, 2, 3, 4, 5]))
    
    seq = [1, 2, 3, 4, 5]
    PyRandomWrapper.shuffle(seq)
    print(seq)
    
    print(PyRandomWrapper.get_random_counter())        # 3
    print(PyRandomWrapper.get_operation_counter())     # {'randint': 1, 'choice': 1, 'shuffle': 1, 'random': 0}
    
    PyRandomWrapper.reset_random_counter() 
    print(PyRandomWrapper.get_random_counter())        # 0
    print(PyRandomWrapper.get_operation_counter())     # {'randint': 0, 'choice': 0, 'shuffle': 0, 'random': 0}