# Old papaper

- it's expensive require million frame of experience




# Idea
* VLn train end-to-end 
    L > Problems  traversable environments

* I/p -> natural language instruction and topological maps
  O/P -> navigation plan

* propose modular approach to VLN using topological maps
   L leverages attentio mechanisms to predict a navigation plan

-Ultize unstructure memory 
   L LSTMS

- Perform navigation
    L mapping
    L planning
    L control

* Cross-Modal Transformer
   
  1.Single Modal Encoder
      - same as standard transformers model
      - i/p  <- map features with trajectory positional encodings
  2.Cross Modal Encoder
      - Query come from one modality
      - Key and value  come from another modality
      - node feature produced by the cross modal encoder ?
      - MLP ?
   3. Trajectory position encoding 
        - predict 1 node token  at a time
        - yeild position encoding  improve planning
* Controller
  
  - phi ~ direction  , p ~ distance
   

cross modal atten-based transfomers
  L compute navigation plans in topological maps
  based on language instructions. 

instruction navigation to generate  an interpretable global navigation plan

# Question:
- what do they mean 
    drop when move during predefined ? 
- metrics maps and tolpological memory 
- relatiomship between instruction and spatial location 
- global navigation plan
- word embedding ?
- circular convolution  

# Problems and solutions

  L two obs from same pose  -> dense topological maps
     L solve by present a node paroma 360 deg

 what's metris map and topological maps ?
  L topological maps such as  graph 
    nodes -> places
    edges ->  environments connectivity or reachability
 
