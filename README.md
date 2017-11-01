# Code

### Behavior Simulator
- Parameters can be adjusted in the init-function
- To generate new behavior, do:
```python
  bg = BehaviorGenerator()
  bg.sample(n=10000, size=5000000)
  with open('<where to save the file>', 'wb') as f:
      pickle.dump([bg.data, bg.duration, bg.gt, bg.etype], f)
```

### Global Pattern Analysis
- Parameters can be reviewed and adjusted in the config file (segmentation/config.py)
- To run GPA, do:
  - load time-series data set as list of lists (each time-series as a separate list)
  - if available also duration information in the same format (if not, set in config collapsed=True)
  - additionally, if available, also groundtruth (only to display metrics)
  - call GPA by 
  ```python
  lbs = GlobalPatternAnalysis(data, duration, gt)
  ```
  
