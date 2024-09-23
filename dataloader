#データセットとデータローダーの取得
train_dataset, train_no_aug, validation_dataset = get_dataset()
train_data_loader = data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
train_NoAug_data_loader = data.DataLoader(train_no_aug, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
validation_data_loader = data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=4)
