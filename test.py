from datasets import load_data

train_dataset, test_dataset, train_dataloader, test_dataloader = load_data('data/cifar10s', batch_size=256,
                                                                           batch_size_test=256, num_workers=4,
                                                                           use_augmentation='base',
                                                                           use_consistency=False, shuffle_train=True,
                                                                           aux_data_filename='/home/jtitor/Downloads/1m.npz',
                                                                           unsup_fraction=0.9, validation=False)

print(len(train_dataset))
for batch in train_dataloader:
    x, y = batch
    print(x.shape)
    
    
