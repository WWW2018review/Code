# Code

### Behavior Simulator (tools/behavior_simulator.py)
- Parameters can be adjusted in the init-function
- To generate new behavior, do:
```python
  bg = BehaviorGenerator()
  bg.sample(n=10000, size=5000000)
  with open('<where to save the file>', 'wb') as f:
      pickle.dump([bg.data, bg.duration, bg.gt, bg.etype], f)
```

### Global Pattern Analysis (main.py)
- Parameters can be reviewed and adjusted in the config file (segmentation/config.py)
- To run GPA, do:
  - load time-series data set as list of lists (each time-series as a separate list)
  - if available also duration information in the same format (if not, set in config collapsed=True)
  - additionally, if available, also groundtruth (only to display metrics)
  - call GPA by 
  ```python
  lbs = GlobalPatternAnalysis(data, duration, gt)
  ```
- The final model is saved in:
```python
save_path = <path to file> + '.pckl'
```
   - with <path to file>: self.cfg.log_path + self.cfg.eval_name (both adjustable in the config)
  
  __An example of how to make use of generated data (behavior simulator) for GPA see main.py.__
