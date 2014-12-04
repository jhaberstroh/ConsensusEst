#!/usr/bin/env python

import pygame, math, sys
import numpy as np, scipy as sp
import random
import numpy.random
import logging
import argparse
from copy import deepcopy

import consensusest.kalman_filter as kalman_filter
import consensusest.sensor as sense

from pygame.locals import *

base = 'consensusest/images'
img_car = base + '/car.png'
img_est = base + '/crosshair.png'
img_sensor = base + '/satellite-dish-icon-hi.png'
 
flags = DOUBLEBUF 

DIRECTION_UP = 0
DIRECTION_DOWN = 1
DIRECTION_LEFT = 2
DIRECTION_RIGHT = 3
DRONE_RANGE=400
DRONE_QTY=4
DRONE_JITTER=50

BLACK = (0,0,0)
MAX_Y = 768
MAX_X = 1024
CAR_INIT_X=100
CAR_INIT_Y=100


class event:
    def __init__(self):
        self.name = "generic"

class event_manager:
    def __init__(self):
        from weakref import WeakKeyDictionary
        self.listeners = WeakKeyDictionary()

    def register_listener(self, listener):
        self.listeners [ listener ] = 1

    def post(self, event):
        for listener in self.listeners.keys():
            listener.notify(event)

class tick_event(event):
    def __init__(self):
        self.name = "tick"

class quit_event(event):
    def __init__(self):
        self.name = "quit"

class car_move_request(event):
    def __init__(self, direction):
        self.name = "request"
        self.direction = direction

class car_move_event(event):
    def __init__(self, car):
        self.name = "move"
        self.car = car

class measurement_event(event):
    def __init__(self, measurement):
        self.name = "meas"
        self.measurement = measurement

class sensor_draw_event(event):
    def __init__(self, sensor):
        self.name = "sens"
        self.sensor = sensor

class keyboard_controller:
    def __init__(self, event_manager):
        self.event_manager = event_manager
        self.event_manager.register_listener(self)
    
    def notify(self, event):
        
        # if not isinstance(event,tick_event):
         #   print ("keyboard notified ")
         #   print event

        
            
        for event in pygame.event.get():
            ev = None
            
            if not hasattr(event, 'key'):
                continue
            
            if event.type == QUIT:
                ev = quit_event()
                
            elif event.type == KEYDOWN:
                if event.key == K_RIGHT:
                    direction = DIRECTION_RIGHT
                    ev = car_move_request(direction)
                elif event.key == K_LEFT:
                    direction = DIRECTION_LEFT
                    ev = car_move_request(direction)
                elif event.key == K_UP:
                    direction = DIRECTION_UP
                    ev = car_move_request(direction)
                elif event.key == K_DOWN:
                    direction = DIRECTION_DOWN
                    ev = car_move_request(direction)
                elif event.key == K_ESCAPE:
                    ev = quit_event()
                    
            self.event_manager.post(ev)


class GameSensorNetwork:        
    def __init__(self, event_manager, mode):
        self.event_manager = event_manager
        self.event_manager.register_listener(self)
        self.converged = 0
        self.old_pos = []
        self.old_est = []
        if mode=='NETWORK':
            logging.debug("Initializing NETWORK mode")
            self.network_init()
        elif mode=='DRONE' or 'DRONE2':
            logging.debug("Initializing DRONE mode")
            self.drone_init()
            self.center_detect=(mode == 'DRONE2')
        else:
            raise ValueError("Invalid type passed to sensor mode: {}"
                    .format(mode))
        logging.debug('Netork after initialization: {}'.
                format(self.net.network))
        
        self.dish_img = pygame.image.load(img_sensor).convert()
        self.crosshair_img = pygame.image.load(img_est).convert()
        self.background_img_cross = pygame.Surface(
                (4.0*self.dish_img.get_width(), 
                4.0*self.dish_img.get_height())).convert()
        self.background_img_cross.fill((0,0,0))
        self.background_img_dish = pygame.Surface(
                (4.0*self.dish_img.get_width(), 
                4.0*self.dish_img.get_height())).convert()
        self.background_img_dish.fill((0,0,0))


        
        self.crosshair_img.set_alpha(64)

    def drone_init(self):
        init_pos = np.matrix([20., 20.]).T
        init_err = np.eye(2) * .01
        self.dyn_model       = np.eye(2)
        self.dyn_noise_model = np.eye(2)
        self.dyn_noise_cov   = np.eye(2) * .03
        # Empty-initialize network and positions
        self.N = 0
        self.net     = sense.LiveSensorNetwork(init_pos, init_err)
        self.net_pos = []
        
        self.drone=True
        self.feedback=True
        self.active=False

        n_drones = DRONE_QTY
        for i in xrange(n_drones):
            rand_pos = np.array([[random.uniform(0, MAX_X), random.uniform(0, MAX_Y)]]).T
            init_belief = np.array([[CAR_INIT_X, CAR_INIT_Y]]).T
            drone = sense.Sensor(model = 
                                 [[random.uniform(0, 1), random.uniform(0, 1)],
                                  [random.uniform(0, 1), random.uniform(0, 1)]],
                                 meas_cov = np.eye(2) * 10. )
            logging.debug("Adding drone {} with {} connections"
                    .format(i, range(self.N)))
            self.add_sensor(drone, pos=rand_pos, connections=range(self.N),
                    x0 = init_belief)
            logging.debug("Added drone {}, N={}.".format(i, self.N))
            self.net.stream_data(init_belief, [i])

    def network_init(self):
        init_pos = np.matrix([20., 20.]).T
        init_err = np.eye(2) * 10.
        self.dyn_model       = np.eye(2)
        self.dyn_noise_model = np.eye(2)
        self.dyn_noise_cov   = np.eye(2) * .03
        # Empty-initialize network and positions
        self.N = 0
        self.net     = sense.LiveSensorNetwork(init_pos, init_err)
        self.net_pos = []

        self.dish_img = pygame.image.load(img_sensor).convert()
        self.crosshair_img = pygame.image.load(img_est).convert()
        self.drone=False
        self.active=True
        
        # Append two sensors to the network
        xm = sense.Sensor([[1., 0.]], [[5.]]) # Measures x
        xm_pos = np.array([20, MAX_Y/2])
        self.add_sensor(xm, xm_pos)
        
        ym = sense.Sensor([[0., 1.]], [[5.]]) # Measures y
        ym_pos = np.array([MAX_X/2, 20])
        self.add_sensor(ym, ym_pos, [0])
        

    def add_sensor(self, new_sensor, pos, connections=[], x0=None):
        self.net.add_sensor(new_sensor, connections, x0=x0)
        pos = np.array(pos)
        self.net_pos.append(pos)
        self.old_pos.append(pos)
        self.old_est.append(self.net[self.N])
        self.N += 1

    def draw(self):
        self.event_manager.post(sensor_draw_event(self))

    def notify(self,event):
        if (isinstance(event,tick_event)):
            self.draw()
        if (isinstance(event,measurement_event)):
            if self.converged >= 10:
                self.active = True
            meas = np.array(event.measurement)
            if not self.drone:
                self.net.stream_data(meas)
                self.net.iterate_filter(dyn_model = self.dyn_model,
                                        dyn_noise_model = self.dyn_noise_model,
                                        dyn_noise_cov = self.dyn_noise_cov)
            if self.drone:
                for i in xrange(self.N):
                    sensor_pos = self.net_pos[i].flatten()
                    target_pos = meas.flatten()
                    target_dist = sensor_pos - target_pos
                    target_mag = np.sqrt(np.sum(np.square(target_dist)))
                    logging.debug("DIST {}: {}".format(i, target_dist))
                    logging.debug("MAG {}: {}".format(i, target_mag))
                    if (target_mag < DRONE_RANGE or not self.active):
                        logging.debug("Drone {} in range".format(i))
                        if not self.center_detect:
                            self.net.stream_data(meas, [i])
                        if self.center_detect:
                            self.net.stream_data( self.net_pos[i], [i])
                        logging.debug(self.net_pos[i].shape)
                        belief = self.net[i]
                        logging.debug(belief.shape)
                        logging.debug(sensor_pos.shape)

                    if self.feedback and self.active:
                        belief = self.net[i]
                        self.old_pos[i] = deepcopy(self.net_pos[i])
                        x_disp = belief[0,0] - sensor_pos[0]
                        y_disp = belief[1,0] - sensor_pos[1]
                        self.net_pos[i][0] += x_disp / 5 + \
                                np.random.normal(scale=DRONE_JITTER)
                        self.net_pos[i][1] += y_disp / 5 + \
                                np.random.normal(scale=DRONE_JITTER)
                self.converged += 1
                self.net.iterate_filter(dyn_model = self.dyn_model,
                        dyn_noise_model = self.dyn_noise_model,
                        dyn_noise_cov = self.dyn_noise_cov)



class car:
    def __init__(self, x, y, speed, direction, turn_speed, max_forward_speed,
            max_reverse_speed,delta,event_manager):
        self.speed = speed
        self.direction = direction
        self.turn_speed = turn_speed
        self.max_forward_speed = max_forward_speed
        self.max_reverse_speed = max_reverse_speed
        self.x = x
        self.y = y
        self.delta = delta
        self.event_manager = event_manager
        self.counter = 0

        self.car_img = pygame.image.load(img_car).convert()
        self.background_image = pygame.Surface((self.car_img.get_width(), self.car_img.get_height())).convert()
        self.background_image.fill((0,0,0))

        self.old_x = x
        self.old_y = y
        self.old_direction = direction
        

    def update_key(self,click_dir):
        if (click_dir == DIRECTION_UP):
            self.speed += self.delta
        elif (click_dir == DIRECTION_DOWN):
            self.speed -= self.delta
        elif (click_dir == DIRECTION_RIGHT):
            self.turn_speed -= self.delta
        elif (click_dir == DIRECTION_LEFT):
            self.turn_speed += self.delta

        if (self.speed > self.max_forward_speed):
            self.speed = self.max_forward_speed
        elif (self.speed < self.max_reverse_speed):
            self.speed = self.max_reverse_speed

        self.direction = self.direction + 0.6 * self.turn_speed
        rad = self.direction * math.pi / 180
        self.x += self.speed * math.sin(rad)
        self.y += self.speed * math.cos(rad)
       
        if self.y < 0:
            self.y = 0 
        elif self.y > MAX_Y:
            self.y = MAX_Y
        if self.x < 0:
            self.x = 0
        elif self.x > MAX_X:
            self.x = MAX_X        
        
    
    def update(self):

        self.old_x = self.x
        self.old_y = self.y
        self.old_direction = self.direction

        self.direction = self.direction + 0.6 * self.turn_speed
        rad = self.direction * math.pi / 180
        self.x += self.speed * math.sin(rad)
        self.y += self.speed * math.cos(rad)

        if (self.speed > self.max_forward_speed):
            self.speed = self.max_forward_speed
        elif (self.speed < self.max_reverse_speed):
            self.speed = self.max_reverse_speed

        if self.y < 0:
            self.y = 0 
        elif self.y > MAX_Y:
            self.y = MAX_Y
        if self.x < 0:
            self.x = 0
        elif self.x > MAX_X:
            self.x = MAX_X        

    def draw(self):
        ev = car_move_event( self )
        self.event_manager.post( ev )

        self.counter += 1

        if self.counter % 3 == 0 :
            ev2 = measurement_event((self.x, self.y))
            self.event_manager.post( ev2 )        

    def notify(self, event):
        if isinstance( event, car_move_request):
            self.update_key(event.direction)
            self.draw()

        if isinstance( event, tick_event):
            self.update()
            self.draw()

class pygame_view:
    def __init__(self, event_manager):
        parser=argparse.ArgumentParser()
        parser.add_argument('-mode', choices=['NETWORK','DRONE', 'DRONE2'], 
                default='NETWORK', help='Choice for mode of operation')
        args = parser.parse_args()

        self.event_manager = event_manager
        self.event_manager.register_listener(self)

        pygame.init()
        self.window = pygame.display.set_mode( (MAX_X,MAX_Y), flags )
        self.window.set_alpha(None)

        pygame.display.set_caption( 'Run Away!' )
        self.background = pygame.Surface( self.window.get_size() ).convert()
        self.background.fill( (0,0,0) )

        self.window.blit( self.background, (0,0) )
        pygame.display.flip()

        self.dish_center = (0,0)
        self.car_center = (0,0)
        self.crosshair_center = (0,0)

        self.dirty_rect_list = []

        self.car = car(x=100, y=100, speed=0, direction=0, turn_speed=0,
                max_forward_speed = 15, max_reverse_speed = 5, delta = 2, 
                event_manager = self.event_manager)
        self.sensor = GameSensorNetwork(event_manager = self.event_manager, 
                mode=args.mode)

        self.counter = 0

    def show_bg(self):
        self.window.blit( self.background, (0,0) )
        pygame.display.flip()


    def show_screen(self):

        def in_range(pos):
            x_ok = pos[0] > 0 and pos[0] < MAX_X
            y_ok = pos[1] > 0 and pos[1] < MAX_Y
            return x_ok and y_ok

        # background blits
#       
        if (self.counter % 10 == 0):
            self.window.blit( self.background, (0,0) )
                
        redraw = pygame.transform.rotate(self.car.background_image,  
                self.car.old_direction)  # draw over this
        redraw_rect = redraw.get_rect()
        redraw_pos = (self.car.old_x, self.car.old_y)    
        redraw_rect.center = redraw_pos
        self.window.blit(redraw, redraw_rect)

        for i in xrange(self.sensor.N):
            dish_bg = self.sensor.background_img_dish
            dish_bg_rect = dish_bg.get_rect()
            old_pos = self.sensor.old_pos[i]
            logging.debug("old_pos array: {}".format(self.sensor.old_pos))
            logging.debug("old_pos: {}".format(old_pos))
            if in_range(old_pos):
                dish_bg_rect.center = (old_pos[0], old_pos[1])
                self.window.blit(dish_bg, dish_bg_rect)

            crosshair_bg = self.sensor.background_img_cross
            crosshair_bg_rect = crosshair_bg.get_rect()
            old_est = self.sensor.old_est[i]
            if in_range(old_est):
                crosshair_bg_rect.center = (old_est[0], old_est[1])
                self.window.blit(crosshair_bg, crosshair_bg_rect)

    # new draws

        for i in xrange(self.sensor.N):
            dish = self.sensor.dish_img
            dish_rect = dish.get_rect() 
            net_pos = self.sensor.net_pos[i]
            if in_range(net_pos):
                dish_rect.center = (net_pos[0], net_pos[1])
                self.window.blit(dish,dish_rect)

        rotated=pygame.transform.rotate(self.car.car_img, self.car.direction)
        rect = rotated.get_rect()
        position = (self.car.x, self.car.y)
        rect.center = position        
        self.window.blit(rotated,rect)
        
        for i in xrange(self.sensor.N):
            crosshair = self.sensor.crosshair_img
            crosshair_rect = crosshair.get_rect()
            est_pos = self.sensor.net[i]
            logging.debug("est_pos, pre: {}".format(est_pos))
            est_pos = (est_pos[0], est_pos[1])
            logging.debug("est_pos, post: {}".format(est_pos))
            if in_range(est_pos):
                crosshair_rect.center = (est_pos[0], est_pos[1])
                self.window.blit(crosshair,crosshair_rect)


        pygame.display.update()

    
    def notify(self, event):
        self.car.notify(event)
        self.sensor.notify(event)
        if (isinstance(event, car_move_event)):
            self.show_screen()
        if (isinstance(event, sensor_draw_event)):
            self.show_screen()

class run_loop:
    def __init__(self, event_manager):
        self.event_manager = event_manager
        self.event_manager.register_listener(self)
        self.running = True

    def run(self):
        while self.running:
            event = tick_event()
            self.event_manager.post( event)

    def notify(self, event):
        if isinstance( event, quit_event ):
            self.running = False

def main():
    ev_manager = event_manager()

    keybd = keyboard_controller(ev_manager)
    spinner = run_loop (ev_manager)

    py_view = pygame_view(ev_manager)

    spinner.run()
 

if __name__ == "__main__":
    #logging.basicConfig(level=logging.DEBUG)
    main()
