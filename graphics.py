import tkinter as tk
from rover import GRID_WIDTH, GRID_HEIGHT
import math

CELL_WIDTH  = 25
CELL_HEIGHT = 25
PADDING     = 20

class playback_positions(tk.Tk):
    
    def __init__(self, true_positions, observed_positions,
                 estimated_positions, estimated_marginals, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.wm_title('Rover Demo')
        self.canvas = tk.Canvas(self,
                                width=GRID_WIDTH*CELL_WIDTH*3 + PADDING*4,
                                height=(GRID_HEIGHT*CELL_HEIGHT + \
                                        PADDING*2),
                                borderwidth=0,
                                highlightthickness=0)
        self.canvas.pack(side='top', fill='both', expand='true')

        self.rect_left    = {}
        self.rect_middle = {}
        self.rect_right = {}
        for column in range(GRID_WIDTH):
            for row in range(GRID_HEIGHT):
                x1 = column * CELL_WIDTH + PADDING
                y1 = row * CELL_HEIGHT
                x2 = x1 + CELL_WIDTH
                y2 = y1 + CELL_HEIGHT
                self.rect_left[row, column] = \
                    self.canvas.create_rectangle(x1, y1, x2, y2,
                                                 fill='white',
                                                 tags='rect_left',
                                                 outline='gray11')

                x1 = GRID_WIDTH*CELL_WIDTH + x1 + PADDING
                x2 = x1 + CELL_WIDTH
                self.rect_middle[row, column] = \
                    self.canvas.create_rectangle(x1, y1, x2, y2,
                                                 fill='white',
                                                 tags='rect_middle',
                                                 outline='gray11')

                x1 = GRID_WIDTH*CELL_WIDTH + x1 + PADDING
                x2 = x1 + CELL_WIDTH
                self.rect_right[row, column] = \
                    self.canvas.create_rectangle(x1, y1, x2, y2,
                                                 fill='white',
                                                 tags='rect_right',
                                                 outline='gray11')

        self.rover_left = self.canvas.create_oval(0, 0, 0, 0, fill='gray80')
        self.rover_left_arrow \
            = self.canvas.create_polygon(0, 0, 0, 0, 0, 0, 0, 0)
        self.canvas.create_text( (GRID_WIDTH*CELL_WIDTH/2.,
                                  GRID_HEIGHT*CELL_HEIGHT + PADDING/3.),
                                 text='True hidden state' )

        self.rover_middle = self.canvas.create_oval(0, 0, 0, 0, fill='gray80')
        self.rover_middle_arrow \
            = self.canvas.create_polygon(0, 0, 0, 0, 0, 0, 0, 0)
        self.canvas.create_text( (GRID_WIDTH*CELL_WIDTH/2. + GRID_WIDTH*CELL_WIDTH +\
                                  PADDING*2,
                                  GRID_HEIGHT*CELL_HEIGHT + PADDING/3.),
                                 text='Observed position' )

        self.rover_right = self.canvas.create_oval(0, 0, 0, 0, fill='gray80')
        self.rover_right_arrow \
            = self.canvas.create_polygon(0, 0, 0, 0, 0, 0, 0, 0)
        self.canvas.create_text( (GRID_WIDTH*CELL_WIDTH/2. + GRID_WIDTH*CELL_WIDTH*2 +\
                                  PADDING*3,
                                  GRID_HEIGHT*CELL_HEIGHT + PADDING/3.),
                                 text='Estimated hidden state' )
        self.time_count = tk.StringVar()
        self.time_count.set('Time Step '+str(0))
        self.time_label = tk.Label(self, textvariable = self.time_count)
        self.time_label.pack()
                    
        self.redraw(true_positions, observed_positions, estimated_positions,
                    estimated_marginals, 800, 0)
       
    def __move_rover(self, rover, rover_arrow, state, horizontal_offset=0):
        if len(state) == 2:
            x, y   = state
            action = None
        else:
            x, y, action = state

        x1 = x * CELL_WIDTH + horizontal_offset
        y1 = y * CELL_HEIGHT 
        x2 = x1 + CELL_WIDTH
        y2 = y1 + CELL_HEIGHT
        self.canvas.coords(rover, x1 + 2, y1 + 2, x2 - 2, y2 - 2)

        cx = (x1 + x2)/2.
        cy = (y1 + y2)/2.
        if action is None:
            self.canvas.coords(rover_arrow, 0, 0, 0, 0, 0, 0,)
        elif action == 'stay':
            self.canvas.coords(rover_arrow,
                               cx - CELL_WIDTH/5., cy - CELL_WIDTH/5,
                               cx - CELL_WIDTH/5., cy + CELL_WIDTH/5,
                               cx + CELL_WIDTH/5., cy + CELL_WIDTH/5.,
                               cx + CELL_WIDTH/5., cy - CELL_WIDTH/5.)
        elif action == 'left':
            self.canvas.coords(rover_arrow,
                               cx - CELL_WIDTH/4., cy,
                               cx - CELL_WIDTH/4., cy,
                               cx + CELL_WIDTH/8., cy - CELL_WIDTH/4.,
                               cx + CELL_WIDTH/8., cy + CELL_WIDTH/4.)
        elif action == 'right':
            self.canvas.coords(rover_arrow,
                               cx + CELL_WIDTH/4., cy,
                               cx + CELL_WIDTH/4., cy,
                               cx - CELL_WIDTH/8., cy - CELL_WIDTH/4.,
                               cx - CELL_WIDTH/8., cy + CELL_WIDTH/4.)
        elif action == 'up':
            self.canvas.coords(rover_arrow,
                               cx, cy - CELL_HEIGHT/4.,
                               cx, cy - CELL_HEIGHT/4.,
                               cx - CELL_WIDTH/4., cy + CELL_HEIGHT/8.,
                               cx + CELL_WIDTH/4., cy + CELL_HEIGHT/8.)
        elif action == 'down':
            self.canvas.coords(rover_arrow,
                               cx, cy + CELL_HEIGHT/4.,
                               cx, cy + CELL_HEIGHT/4.,
                               cx - CELL_WIDTH/4., cy - CELL_HEIGHT/8.,
                               cx + CELL_WIDTH/4., cy - CELL_HEIGHT/8.)   
            
    def move_rover_left(self, state):
        self.__move_rover(self.rover_left, self.rover_left_arrow, state,
                          PADDING)

    def move_rover_middle(self, state):
        self.__move_rover(self.rover_middle, self.rover_middle_arrow, state,
                          GRID_WIDTH * CELL_WIDTH + PADDING*2)

    def move_rover_right(self, state):
        self.__move_rover(self.rover_right, self.rover_right_arrow, state,
                          GRID_WIDTH * CELL_WIDTH*2 + PADDING*3)

    def color_heatmap_grid(self, marginals):
        """
        Color the bottom map, based on the marginal distribution. 
        """
        position_dist = {}
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                position_dist[x, y] = 0.0

        for state, prob in marginals.items():
            position_dist[state[0], state[1]] += prob

        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                color = position_dist[x, y] * 255    
                self.canvas.itemconfigure(self.rect_right[y, x], 
                    fill='#%02x%02x%02x' % (255-int(color), 255-int(color), 255-int(color))) 
                    
    def redraw(self, true_positions, observed_positions, estimated_positions,
               estimated_marginals, delay, time_step):
        if len(true_positions) == 0:
            pass
            #self.destroy()
        else:
            self.canvas.itemconfig('rect_left', fill='white',
                                   outline='gray11')
            self.canvas.itemconfig('rect_middle', fill='white',
                                   outline='gray11')
            self.canvas.itemconfig('rect_right', fill='white',
                                   outline='gray11')
            self.time_count.set('Time Step '+ str(time_step))
            
            if true_positions[0] is not None:
                self.move_rover_left(true_positions[0])
            else:
                self.move_rover_left((-10, -10)) # hide 
                self.canvas.itemconfig('rect_left', fill='red3',
                                       outline='gray11')

            if observed_positions[0] is not None:
                self.move_rover_middle(observed_positions[0])
            else:
                self.move_rover_middle((-10, -10))
                self.canvas.itemconfig('rect_middle', fill='red3',
                                       outline='gray11')

            if estimated_positions[0] is not None:
                self.move_rover_right(estimated_positions[0])
            else:
                self.move_rover_right((-10, -10))
                self.canvas.itemconfig('rect_right', fill='red3',
                                       outline='gray11')

            if estimated_marginals[0] is not None:
                self.color_heatmap_grid(estimated_marginals[0])

            self.after(delay,
                       lambda: self.redraw(true_positions[1:],
                                           observed_positions[1:],
                                           estimated_positions[1:],
                                           estimated_marginals[1:],
                                           delay, time_step+1))