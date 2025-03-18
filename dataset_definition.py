from data_processing import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Data Download")
processed_data, min_max_pos, min_max_state = create_data(device=device)
print("Train - Validation - Test Split")
train_set, validation_set, test_set = train_val_test_split(processed_data)

print(f"Train Set Length : {len(train_set)}\nValidation Set Length : {len(validation_set)}\n"
      f"Test Set Length : {len(test_set)}")
print(f"Minimum and Maximum Position Values : {min_max_pos}\n"
      f"Minimum and Maximum State Values : {min_max_state}")

print("Sets Storage")
save_dataset(data_set=[train_set, min_max_pos, min_max_state], name="train_set")
save_dataset(data_set=[validation_set, min_max_pos, min_max_state], name="validation_set")
save_dataset(data_set=[test_set, min_max_pos, min_max_state], name="test_set")
