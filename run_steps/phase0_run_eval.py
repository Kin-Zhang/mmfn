"""
# Copyright (c) 2021-2022 RAM-LAB
# Authors: kinzhang (qzhangcb@connect.ust.hk)
# Usage: Collect Data or Eval Agent
# Message: All references pls check the readme
"""

from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
from leaderboard.utils.statistics_manager import StatisticsManager
from utils import bcolors as bc
from utils import CarlaServerManager
import hydra
import sys, os, time
import gc
from pathlib import Path


@hydra.main(config_path="config", config_name="collect")
def main(args):
    args.absolute_path = os.environ['CODE_FOLDER']
    args.carla_sh_path = os.path.join(os.environ['CARLA_ROOT'], "CarlaUE4.sh")
    # print(args.absolute_path, args.carla_sh_path)

    # start CARLA =======
    args.trafficManagerPort = args.port + 6000
    if args.if_open_carla:
        server_manager = CarlaServerManager(args.carla_sh_path, port=args.port)
        server_manager.start()

    # select the route files from folder or just file
    if args.routes.split('/')[-1].split('.')[-1] != 'xml':
        print(f'{bc.OKGREEN} =======> input is a folder, start with for loop {bc.ENDC}' )
        route_folder = os.path.join(args.absolute_path, args.routes)
        all_route_files, routes_files = os.listdir(route_folder), []
        for rfile in all_route_files:
            town = rfile.split('_')[1].capitalize()
            if town in list(args.towns):
                routes_files.append(os.path.join(args.absolute_path, args.routes, rfile))
                
    else:
        routes_files = [os.path.join(args.absolute_path, args.routes)]

    # config init =============> make all path with absolute
    args.scenarios = os.path.join(args.absolute_path,args.scenarios)
    args.agent     = os.path.join(args.absolute_path, args.agent)

    if 'data_save' in args.agent_config:
        origin_data_folder = args.agent_config.data_save
        args.agent_config.data_save = os.path.join(args.absolute_path, origin_data_folder)
        Path(args.agent_config.data_save).mkdir(exist_ok=True, parents=True)
    
    if 'model_path' in args.agent_config:
        args.agent_config.model_path = os.path.join(args.absolute_path, args.agent_config.model_path)

    print('-'*20 + "TEST Agent: " + bc.OKGREEN + args.agent.split('/')[-1] + bc.ENDC + '-'*20)
    for rfile in routes_files:
        # check if there are many route files
        if len(routes_files) >1:
            args.agent_config.town = rfile.split('/')[-1].split('_')[1].capitalize()
            # make sure have route folder
            if 'data_save' in args.agent_config:
                args.agent_config.data_save = os.path.join(args.absolute_path, origin_data_folder, rfile.split('/')[-1].split('.')[0][7:].capitalize())
                Path(args.agent_config.data_save).mkdir(exist_ok=True, parents=True)

        args.routes = rfile
        route_name = args.routes.split('/')[-1].split('.')[0]
        args.checkpoint = args.agent.split('/')[-1].split('.')[0] + '.json'

        # make sure that folder is exist
        data_folder = os.path.join(args.absolute_path,'data/results')
        Path(data_folder).mkdir(exist_ok=True, parents=True)
        args.checkpoint = os.path.join(data_folder,f'{route_name}_{args.checkpoint}')

        if os.path.exists(args.checkpoint) and not args.resume:
            print(f"It will overwrite the things! {bc.WARNING}ATTENTION {args.checkpoint}{bc.ENDC}")
        elif args.resume:
            print(f"{bc.UNDERLINE}Contiune the route from file{bc.ENDC}: {args.checkpoint}")
        else:
            print(f"Create the result to: {args.checkpoint}")
            
        # run official leaderboard ====>
        scenario_manager = StatisticsManager()
        leaderboard_evaluator = LeaderboardEvaluator(args, scenario_manager)
        leaderboard_evaluator.run(args)
        leaderboard_evaluator.__del__() # important to run this one!!!!!
        # run official leaderboard ====>

        # kill CARLA ===> attention will kill all CARLA!
        # server_manager.stop()
        
if __name__ == '__main__':
    
    start_time = time.time()
    main()
    print('clean memory on no.', gc.collect(), "Uncollectable garbage:", gc.garbage)
    print(f"{bc.OKGREEN}TOTAL RUNNING TIME{bc.ENDC}: --- %s hours ---" % round((time.time() - start_time)/3600,2))