from PIL import Image

'''
Temporary "Fixed" size for image - 10/10/2025
'''

# Resize letter box
def Letterbox_resize(image, target_img_size=(480, 480)):
    
    # Read image size
    original_width, original_height = image.size  
    ratio = min(target_img_size[0] / original_width, target_img_size[1] / original_height) # Resize Ratio
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)    

    # Create a new image with padding
    new_image = Image.new('RGB', target_img_size, (0, 0, 0))
    padding_x = (target_img_size[0] - new_width) // 2
    padding_y = (target_img_size[1] - new_height) // 2
    new_image.paste(resized_image, (padding_x, padding_y))

    return new_image, ratio, (padding_x, padding_y)