import os

def get_items(identity, data):
    items = [x for x in data if identity in x['likely']]
    return items

def extract_distance(item_info, identity):
    return item_info['likely'][identity]

def sort_by_distance(items_info, identity):
    items_info.sort(key=lambda k: extract_distance(k, identity))

def get_thumb_name(identity):
    return os.listdir(f'./static/{identity}')[0]

def get_item_info(item_data, item_id):
    if item_data[int(item_id)]['key'] == item_id:
        return item_data[int(item_id)]
    else:
        for item in item_data:
            if item['key'] == item_id:
                return item