import json
import itertools
import os

def read_config():
    with open("master_config.json", 'r') as json_file: 
      data = json.load(json_file)
    return data

def main():
    config = read_config()
    try:
      assert os.path.exists("cfg_temp/")
    except:
      os.makedirs('cfg_temp/')
    i=0
    for param_vals in itertools.product(*config.values()):
        cfg = dict(zip(config.keys(), param_vals))
        config_struct ={
            "alpha" : cfg["alpha"],
            "gamma" : cfg["gamma"],
            "epsilon" : cfg["epsilon"],
            "N_0" : cfg["N_0"],
            "numEpisodes" : cfg["numEpisodes"],
            "stepsPerEpisode" : cfg["stepsPerEpisode"],
            "n_planning_steps" : cfg["n_planning_steps"],
        }
        jsonstr = json.dumps(config_struct)
        w_name = 'cfg_temp/'+ str(i) + '.json'
        i = i+1
        with open(w_name, "w") as outfile: 
            outfile.write(jsonstr) 


if __name__ == '__main__':
    main()