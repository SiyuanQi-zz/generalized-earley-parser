"""
Created on Jan 30, 2018

@author: Siyuan Qi

Description of the file.

"""

subactivities = [
    'null',
    'fetch_from_fridge', 'put_back_to_fridge', 'prepare_food', 'microwaving', 'fetch_from_oven',
    'pouring', 'drinking', 'leave_kitchen', 'fill_kettle', 'plug_in_kettle', 'move_kettle',
    'reading', 'walking', 'leave_office', 'fetch_book', 'put_back_book', 'put_down_item',
    'take_item', 'play_computer', 'turn_on_monitor', 'turn_off_monitor'
]

subactivity_index = dict()
for s in subactivities:
    subactivity_index[s] = subactivities.index(s)


def main():
    pass


if __name__ == '__main__':
    main()
