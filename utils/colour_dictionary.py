def get_colour_list():
    "These are the eleven basic colours (in english) plus the background"
    color_list = [[0, 153, 153],
                  [0, 0, 0],
                  [0, 0, 255],
                  [102, 51, 0],
                  [127.5, 127.5, 127.5],
                  [0, 255, 0],
                  [255, 153, 0],
                  [204, 153, 178.5],
                  [178.5, 0, 178.5],
                  [255, 0, 0],
                  [255, 255, 255],
                  [255, 255, 0]]

    color_names = {0: 'backgroun',
                 1: 'black',
                 2: 'blue',
                 3: 'brown',
                 4: 'grey',
                 5: 'green',
                 6: 'orange',
                 7: 'pink',
                 8: 'purple',
                 9: 'red',
                 10: 'white',
                 11: 'yellow'}
    return color_list, color_names