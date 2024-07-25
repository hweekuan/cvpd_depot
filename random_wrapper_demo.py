from control_random import PyRandomWrapper

def test_python_random():
    """ Test the python controlled random class
    """    
    print('Python Random Testing: **************************************************************************\n')
    
    # Random seed --------------------------------------------------------------------------------------------
    ## If not specified, the seed is automatically generated, and this value is returned as None
    print("Random Seed if not specified:", PyRandomWrapper.get_seed())
    assert PyRandomWrapper.get_seed() is None, "The seed should be None if not specified"
    
    ## Set seed
    mySeed = 42
    PyRandomWrapper.seed(mySeed)
    print("Random Seed is set:", PyRandomWrapper.get_seed())
    assert PyRandomWrapper.get_seed() == mySeed, "The seed should be set to 42"
    
    print("Test for seed passed! \n")
    
    # Random operation --------------------------------------------------------------------------------------
    ## Random integer
    print("Random number:", PyRandomWrapper.random())
    
    ## Random choice from a list 
    print("Random choice:", PyRandomWrapper.choice([1, 2, 3, 4, 5]))
    
    ## Random shuffle
    seq = [1, 2, 3, 4, 5]
    PyRandomWrapper.shuffle(seq)
    print("Random shuffle the sequence [1, 2, 3, 4, 5]:", seq)
    
    # Other random operations can be used: randint, choices, uniform, etc. 
    
    # Get random counter -------------------------------------------------------------------------------------
    ## Total use of random
    print("Total use of random:", PyRandomWrapper.get_random_counter())
    assert PyRandomWrapper.get_random_counter() == 3, "The total use of random should be 3"
    print("Test for counter passed! \n")

    ## Total use of each random operation
    print("Total use of each random operation:", PyRandomWrapper.get_operation_counter())
    
    ## Reset random counter
    PyRandomWrapper.reset_random_counter()
    print("Reset random counter:", PyRandomWrapper.get_random_counter())
    print("Reset random operation counter:", PyRandomWrapper.get_operation_counter())   
    assert PyRandomWrapper.get_random_counter() == 0, "The total use of random should be 0"
    print("Test for counter reset passed!")
    

def test_singleton():
    """ Test the singleton implemenation of the controlled random class, 
        to ensure that the same instance is returned and the counter is correctly updated.
    """
    print('\nSingleton Testing: **************************************************************************\n')
    # Reset random counter
    PyRandomWrapper.reset_random_counter()
    print("Random counter is reset!")

    # Create an instance of the controlled random class (only for testing purposes) ------------------------------
    # In fact, the instance creation is not necessary, all the methods are static so they can be called directly from the class (see test_python_random())
    instance_1 = PyRandomWrapper()
    
    # Set seed    
    instance_1.seed(42)
    print("Random Seed:", instance_1.get_seed())
    
    # Perform random operations
    print("Random integer: ", instance_1.randint(0, 10))
    print("Random 2 numbers from [1, 2, 3, 4, 5]: ", instance_1.choices([1, 2, 3, 4, 5], 2))
    
    # Print out random counter
    instance1_rand_counter = instance_1.get_random_counter()
    print("Random counter: ", instance1_rand_counter)
    print("Random operation counter: ", instance_1.get_operation_counter())
    assert instance1_rand_counter == 2, "The random counter should be 2"
    
    # Create another instance of the controlled random class -----------------------------------------------------
    # It should return the same instance as the previous one
    instance_2 = PyRandomWrapper()

    # Set seed
    instance_2.seed(123)
    print("Random Seed of instance 2:", instance_2.get_seed())
    print("Random Seed of instance 1:", instance_1.get_seed())  # The seed should be changed to the new value, as they are the same instance
    assert instance_2.get_seed() == 123, "The seed should be set to 123"
    assert instance_1.get_seed() == 123, "The seed should be the same for both instances"
    print("Test for singleton seed passed!")

    # Print out random counter
    instance2_rand_counter = instance_2.get_random_counter()  
    print("Random counter: ", instance2_rand_counter) # The counter should be same as the previous instance 
    print("Random operation counter: ", instance_2.get_operation_counter())
    assert instance1_rand_counter == instance2_rand_counter, "The random counter should be the same for both instances"
    print("Random counter for both instances are the same!\n")
    
    # Perform random operations
    print("Random integer: ", instance_2.randint(0, 10))
    
    # Print out random counter
    instance2_new_rand_counter = instance_2.get_random_counter()
    print("Random counter of instance 2 after 1 more random operation: ", instance2_new_rand_counter)
    print("Random counter of instance 1: ", instance_1.get_random_counter())
    
    assert instance2_new_rand_counter == instance_1.get_random_counter(), "The random counter should be the same for both instances"
    print("Random counter for both instances are the same!")
    print("Test for singleton random counter passed!\n")
    
    # Reset random counter
    instance_2.reset_random_counter()
    print("Random counter is reset!")
    print("All tests passed!")
    

    
def main():
    test_python_random()
    test_singleton()
    
if __name__ == '__main__':
    main()