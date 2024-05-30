

get a list of all files in the various subfolders using glob
iterate through them, providing the function "processImage" to each filename. The processImage filename will split the filename into various components, that will be saved in a csv file. That CSV file will have the first column as the original full filename, and then the rest of the columns will be "genotype", "**condition**", "date", "block", "
processImage:

    First extract the metadata:
        The first subfolder name include the plant genotype and the condition, separated by '-'.
        The next subfolder is the date, in the format "YY-MM-DD".
        The next subfolder is the block number, in the format "Block 3"
        Within that folder is the filename like: "024 WT-500-uE-3 19k.tif"
        We want the first number which represents the layer number, the rep number that is the -3 in this case, and the 19k.
        Return this metadata as a dict. 

    Now, we load the tif image and then save it as a png in a raw_image directory, named as:
        date_genotype_condition_block_rep_layer.png

    The image_index file in that directory should be updated, with the dict appended as a line of the csv and then save it again. 