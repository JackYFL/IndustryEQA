import json
import sys

def process_json(input_file, output_file, ids_to_remove):
    """
    Read a JSON file, remove items with question_ids in the provided list,
    and reassign question_ids sequentially.
    """
    try:
        # Read the JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if it's a list of items or an object with items property
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'items' in data:
            items = data['items']
        else:
            print("JSON structure not recognized. Expected a list or an object with 'items' property.")
            return False
        
        print(f"Original number of items: {len(items)}")
        
        # Remove items with question_ids in the list
        filtered_items = [item for item in items if 'question_id' not in item or item['question_id'] not in ids_to_remove]
        
        print(f"Number of items after removal: {len(filtered_items)}")
        print(f"Removed {len(items) - len(filtered_items)} items")
        
        # Reassign question_ids sequentially
        for i, item in enumerate(filtered_items, 1):
            if 'question_id' in item:
                item['question_id'] = i
        
        # Prepare output data in the same structure as input
        if isinstance(data, list):
            output_data = filtered_items
        else:
            output_data = data.copy()
            output_data['items'] = filtered_items
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully processed JSON and saved to {output_file}")
        return True
        
    except json.JSONDecodeError:
        print(f"Error: {input_file} is not a valid JSON file")
        return False
    except Exception as e:
        print(f"Error processing JSON: {str(e)}")
        return False

if __name__ == "__main__":

    
    input_file = "./results/results/large_all_new.json"
    output_file = "./results/results/large_all_new_new.json"
    
    # The list of question_ids to remove
    # ids_to_remove = [10, 27, 28, 29, 31, 44, 78, 83, 89, 129, 147, 156, 160, 177, 180, 182, 184, 185, 186, 187, 192, 201, 203, 204, 207, 209, 210, 211, 212, 221, 228, 229, 233, 234, 235, 236, 237, 252, 254, 255, 256, 270, 274, 283, 289, 300, 302, 307, 310, 328, 329, 330, 332, 345, 348, 349, 351, 353, 354, 356, 357, 363, 368, 370, 371, 373, 374, 375, 376, 381, 382, 383, 385, 386, 387, 388, 391, 393, 395, 405, 407, 409, 432, 436, 440, 441, 450, 451, 470, 475, 482, 487, 489, 495, 502, 513, 521, 522, 524, 538, 542, 555, 557, 567, 568, 569, 578, 585, 606, 607, 609, 620, 623, 637, 647, 665, 669, 693, 695, 705, 707, 708, 710, 713, 714, 726, 730, 731, 734, 749, 750, 751, 752, 753, 754, 774, 787, 790, 794, 795, 796, 798, 800, 802, 806, 810, 812, 824, 828, 830, 832, 833, 834, 835, 844, 852, 863, 868, 878, 880, 883, 884, 886, 888, 889, 890, 913, 930, 932, 933, 935, 937, 940, 949, 959, 960, 963, 972, 983, 985, 986, 990, 994, 995, 998, 1000, 1013, 1014, 1015, 1016, 1018, 1027, 1030, 1037, 1041, 1045, 1046, 1050, 1078, 1088, 1095, 1127, 1129, 1130, 1131, 1133, 1137, 1138, 1146, 1151, 1177, 1185, 1187, 1198, 1207, 1211, 1212, 1213, 1214, 1216, 1223, 1229, 1234, 1236]
    ids_to_remove = [6, 8, 13, 25, 52, 54, 56, 58, 62, 65, 76, 78, 81, 83, 87, 95, 98, 108, 112, 114, 115, 116, 124, 132, 133, 135, 138, 142, 146, 147, 148, 154, 156, 158, 160, 161, 166, 171, 174, 177, 185, 188, 189, 195, 198, 202, 203, 206, 207, 215, 217, 225, 226, 228, 229, 230, 231, 233, 239, 252, 254, 256, 262, 266, 267, 274, 276, 284, 286, 298, 302, 304, 306, 308, 310, 311, 313, 320, 321, 322, 326, 330, 332, 333, 334, 338, 351, 352, 354, 358, 360, 365, 366, 374, 375, 376, 378, 380, 381, 391, 392, 393, 394, 402, 406, 410, 414, 417, 418, 420, 423, 424, 429, 432, 435, 436, 437, 441, 444, 445, 447, 448, 450, 454, 464, 467, 468, 475, 476, 479, 480, 481, 485, 486, 488, 490, 492, 494, 497, 500, 509, 511, 516]
    process_json(input_file, output_file, ids_to_remove)