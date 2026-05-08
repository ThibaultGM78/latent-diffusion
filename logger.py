import csv
import os

def log_loss_to_csv(csv_path, epoch, train_loss, val_loss=None):
    """
    Saves training and validation losses to a CSV file.
    Creates the file and headers if it doesn't exist.
    """
    file_exists = os.path.isfile(csv_path)
    
    
    # Open in 'append' mode so we don't overwrite previous epochs
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write the header only if the file is brand new
        if not file_exists:
            if val_loss is not None:
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
            else:
                writer.writerow(['epoch', 'train_loss'])
        
        # Write the data for the current epoch
        if val_loss is not None:
            writer.writerow([epoch, train_loss, val_loss])
        else:
            writer.writerow([epoch, train_loss])