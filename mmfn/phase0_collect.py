"""
# Copyright (c) 2021-2022 RAM-LAB
# Authors: kinzhang (qzhangcb@connect.ust.hk)
# Usage: Collect Data

For **collect data**, note that you have to launch carla first!
./scripts/launch_carla.sh 1 2000

"""
from copy import deepcopy
from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
from leaderboard.utils.statistics_manager import StatisticsManager
from mmfn.utils import *
import hydra
import sys, os
import datetime
import gc
import time

class ScenarioRunner():
    def __init__(self, args, scenario_class, scenario, checkpoint='simulation_results.json', town=None, port=1000, tm_port=1002):
        args = deepcopy(args)
        
        # Inject args
        args.scenario_class     = scenario_class
        # 会检查和 route 头文件读出来的是否一致：config = route_indexer.next() if config.town == self.town:
        args.town               = town
        args.port               = port
        args.trafficManagerPort = tm_port
        args.scenarios          = scenario
        args.checkpoint         = checkpoint
        args.record             = ''
        args.agent  = args.absolute_path + args.agent
        args.agent_config.data_save = args.absolute_path + args.agent_config.data_save
        self.runner = LeaderboardEvaluator(args, StatisticsManager())
        self.args = args

    def run(self):
        return self.runner.run(self.args)

@hydra.main(config_path="../config", config_name="ncollect")
def main(args):
    # config init
    scenario = args.absolute_path + args.scenarios
    scenario_class = args.scenario_class
    route_folder = args.absolute_path + args.routes
    all_routes = os.listdir(route_folder)
    port = args.port
    tm_port = args.port + 6000 + args.trafficManagerPort
    data_save = args.agent_config.data_save
    for route_file in all_routes:
        town = route_file.split('_')[1].capitalize()
        if route_file.split('_')[2] == 'tiny.xml':
            continue
        if town in list(args.towns):
            print('-'*20 + "TEST Agent: " + bcolors.OKGREEN + args.agent + bcolors.ENDC + '-'*20)
            args.routes = route_folder + route_file
            
            route_name = route_file.split('.')[0][7:].capitalize()
            args.agent_config.data_save = data_save + '/' + route_name
            # set time
            now = datetime.datetime.now()
            string = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute)))
            if eval(args.checkpoint) is None:
                args.checkpoint = args.agent.split('/')[-1].split('_')[0] + '.json'
            checkpoint = f'{args.absolute_path}results/{route_name}_{string}_{args.checkpoint}'
            args.agent_config.town = route_name
            runner = ScenarioRunner(args, scenario_class, scenario, checkpoint=checkpoint, town=town, port=port, tm_port=tm_port)
            runner.run()
            print('Clean memory on no.', gc.collect(), "Uncollectable garbage:", gc.garbage)

if __name__ == '__main__':
    
    start_time = time.time()

    try:
        main()
    except Exception as e:
        print("> {}\033[0m\n".format(e))
        print(bcolors.FAIL + "There is a error or close by users"+ bcolors.ENDC, 'clean memory', gc.collect())
        sys.exit()

    print('clean memory on no.', gc.collect(), "Uncollectable garbage:", gc.garbage)
    print(f"{bcolors.OKGREEN}TOTAL RUNNING TIME{bcolors.ENDC}: --- %s hours ---" % round((time.time() - start_time)/3600,2))
