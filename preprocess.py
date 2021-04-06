import pandas as pd
import numpy as np
class Preprocess:
    def __init__(self,data):
        self.data=data
        self.play_train_df=None
        self.train_df=None   


    
    def forward_final(self):
        self.forward_standard()
        self.forward_dir()
        self.forward_1()
        self.forward_rusher()
        self.forward_relative()
        self.forward_box()
        self.train_df['Team'] = self.train_df['Team'].map({'away':1,'home':0})
        def f(x):
            d = {}
            l=[]
            default_x=x[x['IsBallCarrier']]['X_std']
            team=x[x['IsBallCarrier']]['Team']
            l.append(*team)

            #d['YardLine']= [100-x[x.loc.ToLeft][x.YardLine>50] if x[x.ToLeft][x.YardLine>50] else x.YardLine]
            #d['YardLine']=x.reset_index().loc[0,['YardLine']]



            if 0 in l:
              d['X_difference_max']=x[x['Team']==0]['X_std'].max() - default_x
              d['X_difference_min']=x[x['Team']==0]['X_std'].min() - default_x

              default_y=x[x['IsBallCarrier']]['Y_std']

              d['Y_difference_max']=x[x['Team']==0]['Y_std'].max() - default_y
              d['Y_difference_min']=x[x['Team']==0]['Y_std'].min() - default_y

              d['opponent_max_x']=x[x['Team']==1]['X_std'].max()
              d['opponent_max_y']=x[x['Team']==1]['Y_std'].max()

              d['home_max_x']=x[x['Team']==0]['X_std'].max()

              Y_large=x.sort_values(by=['X_std'],ascending=False).reset_index().loc[0,['Y_std']]

              first=(Y_large-x['Y_std'].std() <= d['opponent_max_y'])
              second=(d['opponent_max_y']<= Y_large+x['Y_std'].std())
              d['Near']=all([first.Y_std,second.Y_std])



            elif 1 in l:
              d['X_difference_max']=x[x['Team']==1]['X_std'].max() - default_x
              d['X_difference_min']=x[x['Team']==1]['X_std'].min() - default_x

              default_y=x[x['IsBallCarrier']]['Y_std']

              d['Y_difference_max']=x[x['Team']==1]['Y_std'].max() - default_y
              d['Y_difference_min']=x[x['Team']==1]['Y_std'].min() - default_y

              d['opponent_max_x']=x[x['Team']==0]['X_std'].max()
              d['opponent_max_y']=x[x['Team']==0]['Y_std'].max()
              d['home_max_x']=x[x['Team']==1]['X_std'].max()

              Y_large=x.sort_values(by=['X_std'],ascending=False).reset_index().loc[0,['Y_std']]
              first=(Y_large-x['Y_std'].std() <= d['opponent_max_y'])
              second=(d['opponent_max_y']<= Y_large+x['Y_std'].std())
              d['Near']=all([first.Y_std,second.Y_std])





            else:
              pass 

            return pd.Series(d,index=['X_difference_max','Y_difference_max','X_difference_min','Y_difference_min','opponent_max_x','opponent_max_y','home_max_x'],dtype=float)
        self.train_df=self.train_df[['PlayId', 'X_std','IsBallCarrier','Team','Y_std','YardLine','IsOnOffense','ToLeft']].reset_index(drop=True).groupby('PlayId').apply(f)
        final_data=self.play_train_df.merge(self.train_df,on='PlayId')
        final_data.drop(['PlayId'],axis=1,inplace=True)
        
        self.data=None
        self.play_train_df=None
        self.train_df=None  
        
     
        return final_data
    
        
    def forward_standard(self):
        # fill some NAs
        self.data.loc[self.data.S.isnull(),'S'] = 2.6
        self.data.loc[self.data.A.isnull(),'A'] = 1.6  




        #add a moving left flag and get rusher
        self.data['ToLeft'] = self.data['PlayDirection'] == 'left'
        self.data['IsBallCarrier'] = self.data['NflId'] == self.data['NflIdRusher']

        #correct some naming differences
        self.data.loc[self.data['VisitorTeamAbbr'] == "ARI", 'VisitorTeamAbbr'] = 'ARZ'
        self.data.loc[self.data['HomeTeamAbbr'] == "ARI", 'HomeTeamAbbr'] = 'ARZ'

        self.data.loc[self.data['VisitorTeamAbbr'] == "BAL", 'VisitorTeamAbbr'] = 'BLT'
        self.data.loc[self.data['HomeTeamAbbr'] == "BAL", 'HomeTeamAbbr'] = 'BLT'

        self.data.loc[self.data['VisitorTeamAbbr'] == "CLE", 'VisitorTeamAbbr'] = 'CLV'
        self.data.loc[self.data['HomeTeamAbbr'] == "CLE", 'HomeTeamAbbr'] = 'CLV'

        self.data.loc[self.data['VisitorTeamAbbr'] == "HOU", 'VisitorTeamAbbr'] = 'HST'
        self.data.loc[self.data['HomeTeamAbbr'] == "HOU", 'HomeTeamAbbr'] = 'HST'

        #work out who is on offense
        self.data['TeamOnOffense'] = np.where(self.data['PossessionTeam'] == self.data['HomeTeamAbbr'], 'home', 'away')
        self.data['IsOnOffense'] = self.data['TeamOnOffense'] == self.data['Team']

        #get the standardized yards
        self.data['YardsFromOwnGoal'] = np.where(self.data['FieldPosition'] == self.data['PossessionTeam'], self.data['YardLine'], 50 + (50 - self.data['YardLine']))
        self.data['YardsFromOwnGoal'] = np.where(self.data['YardLine'] == 50, 50, self.data['YardsFromOwnGoal'])


        #get standardized X,Y coordinates
        self.data['X_std'] = np.where(self.data['ToLeft'], 120 - self.data['X'], self.data['X']) - 10
        self.data['Y_std'] = np.where(self.data['ToLeft'], 160 / 3  - self.data['Y'], self.data['Y'])

        #get standardized direction
        self.data['Dir_std_1'] = np.where( (self.data['Dir'] < 90) & self.data['ToLeft'], self.data['Dir'] + 360, self.data['Dir'])
        self.data['Dir_std_1'] = np.where( (self.data['Dir'] > 270) & ~self.data['ToLeft'], self.data['Dir'] - 360, self.data['Dir_std_1'])
        #This is used to flip the players directtions from right to left in 180 degree out of phase
        self.data['Dir_std_2'] = np.where(self.data['ToLeft'], self.data['Dir_std_1'] - 180, self.data['Dir_std_1'])

        self.data['Dir_std_2'] = self.data['Dir_std_2']*np.pi/180

        #fill any na's in the standardized direction that is player is moving in straight line
        self.data.loc[self.data.Dir_std_2.isnull(),'Dir_std_2'] = np.pi/2
    
    
    def forward_dir(self):
        
    
        def x_component_of_dir(row):
            if 0< row['Dir_std_2'] <= np.pi/2:
                return np.cos(row['Dir_std_2'])
            if np.pi/2< row['Dir_std_2'] <= np.pi:
                return 1*np.cos(np.pi-row['Dir_std_2'])
            if row['Dir_std_2'] > np.pi:
                return -1*np.cos(row['Dir_std_2']-np.pi)
            return 1
    
        
        def y_component_of_dir(row):
            if 0< row['Dir_std_2'] <= np.pi/2:
                return np.sin(abs(row['Dir_std_2']))
            if np.pi/2< row['Dir_std_2'] <= np.pi:
                return 1*np.sin(abs(np.pi-row['Dir_std_2']))
            if row['Dir_std_2'] > np.pi:
                return -1*np.sin(abs(row['Dir_std_2']-np.pi))
            return 1
        #Now calculate X and Y component of Dir
        self.data['x_component_of_dir']=self.data.apply(x_component_of_dir,axis=1)
        self.data['y_component_of_dir']=self.data.apply(y_component_of_dir,axis=1)
        
    def forward_1(self):
        
           #Now calculate X and Y component of Speed
        self.data['x_component_of_dir_speed']=self.data['S']*self.data['x_component_of_dir']
        self.data['y_component_of_dir_speed']=self.data['S']*self.data['y_component_of_dir']



        #Now calculate X and Y component of Speed after 0.5 seconds
        self.data['x_component_of_dir_distance_after_0.5sec']=self.data['X_std']+self.data['x_component_of_dir_speed']*0.5
        self.data['y_component_of_dir_distance_after_0.5sec']=self.data['Y_std']+self.data['y_component_of_dir_speed']*0.5

        #Now calculate X and Y component of Speed after 1 seconds
        self.data['x_component_of_dir_distance_after_1sec']=self.data['X_std']+self.data['x_component_of_dir_speed']
        self.data['y_component_of_dir_distance_after_1sec']=self.data['Y_std']+self.data['y_component_of_dir_speed']


        #Drop some columns which are not useful furthur
        #drop_cols = ['X','Y','Dir','PossessionTeam','FieldPosition','NflIdRusher','PlayDirection','HomeTeamAbbr','VisitorTeamAbbr','TeamOnOffense','Dir_std_1']

        #self.data.drop(drop_cols, axis = 1, inplace = True)
        self.train_df=self.data
    
    def forward_rusher(self):
        
        # work on the play_info df
        play_info_cols = ['PlayId','Distance','YardsFromOwnGoal']

        self.play_train_df = self.data[play_info_cols]
        self.play_train_df = self.play_train_df.drop_duplicates()

        # fill some na's with average value
        self.play_train_df.loc[self.play_train_df.Distance.isnull(), 'Distance'] = 8.3 
        self.play_train_df.loc[self.play_train_df.YardsFromOwnGoal.isnull(), 'YardsFromOwnGoal'] = 50 

        #get rusher to game_info
        rushers = self.data.loc[self.train_df.IsBallCarrier,['PlayId','X_std','Y_std','x_component_of_dir_speed','y_component_of_dir_speed','x_component_of_dir_distance_after_0.5sec','y_component_of_dir_distance_after_0.5sec','S','A','Dir_std_2','x_component_of_dir_distance_after_1sec','y_component_of_dir_distance_after_1sec']]
        rushers.columns = ['PlayId','X_std_rush','Y_std_rush','x_component_of_dir_speed_rush','y_component_of_dir_speed_rush','x_component_of_dir_distance_after_0.5sec_rush','y_component_of_dir_distance_after_0.5sec_rush','S_Rush','A_Rush','Dir_std_2_Rush','x_component_of_dir_distance_after_1sec_rush','y_component_of_dir_distance_after_1sec_rush']
        self.play_train_df = self.play_train_df.merge(rushers,how = 'left', on = 'PlayId' )

        self.train_df = self.train_df.merge(rushers,how = 'left', on = 'PlayId' )
   
    
    def forward_relative(self):
        #Relative difference between rusher and other players
        self.train_df['X_rel_rush'] = self.train_df['X_std_rush'] - self.train_df['X_std']
        self.train_df['Y_rel_rush'] = self.train_df['Y_std_rush'] - self.train_df['Y_std']

        #Relative difference between rusher and other player X component
        #train_df['X_rel_rush_speed'] = train_df['x_component_of_dir_speed_rush'] - train_df['x_component_of_dir_speed']
        #train_df['Y_rel_rush_speed'] = train_df['y_component_of_dir_speed_rush'] - train_df['y_component_of_dir_speed']


        #Relative difference between rusher and other player X component
        self.train_df['X_rel_rush_0.5sec_distance'] = self.train_df['x_component_of_dir_distance_after_0.5sec_rush'] - self.train_df['x_component_of_dir_distance_after_0.5sec']
        self.train_df['Y_rel_rush_0.5sec_distance'] = self.train_df['y_component_of_dir_distance_after_0.5sec_rush'] - self.train_df['y_component_of_dir_distance_after_0.5sec']

        #Relative difference between rusher and other player X component
        self.train_df['X_rel_rush_1sec_distance'] = self.train_df['x_component_of_dir_distance_after_1sec_rush'] - self.train_df['x_component_of_dir_distance_after_1sec']
        self.train_df['Y_rel_rush_1sec_distance'] = self.train_df['y_component_of_dir_distance_after_1sec_rush'] - self.train_df['y_component_of_dir_distance_after_1sec']

        #Now calculate euclidian distance between rusher and each player
        self.train_df['euclidian_home']= np.sqrt( self.train_df['X_rel_rush']**2 + self.train_df['Y_rel_rush']**2 )
        self.train_df['euclidian_opponent']=self.train_df.apply(lambda row: row.euclidian_home if not row.IsOnOffense else np.NaN, axis = 1)
        #after 0.5 seconds
        self.train_df['euclidian_home_halfsec']= np.sqrt( self.train_df['X_rel_rush_0.5sec_distance']**2 + self.train_df['Y_rel_rush_0.5sec_distance']**2 )
        self.train_df['euclidian_opponent_halfsec']=self.train_df.apply(lambda row: row.euclidian_home_halfsec if not row.IsOnOffense else np.NaN, axis = 1)
        #after 1 second

        self.train_df['euclidian_home_1sec']= np.sqrt( self.train_df['X_rel_rush_1sec_distance']**2 + self.train_df['Y_rel_rush_1sec_distance']**2 )
        self.train_df['euclidian_opponent_1sec']=self.train_df.apply(lambda row: row.euclidian_home_1sec if not row.IsOnOffense else np.NaN, axis = 1)


        mean_def_dist   = self.train_df[['PlayId', 'euclidian_opponent']].groupby('PlayId').agg(DistToRusherDefMean = ('euclidian_opponent',np.nanmean))
        min_def_dist    = self.train_df[['PlayId', 'euclidian_opponent']].groupby('PlayId').agg(DistToRusherDefMin = ('euclidian_opponent',np.nanmin))

        mean_def_dist_new   = self.train_df[['PlayId', 'euclidian_opponent_1sec']].groupby('PlayId').agg(DistToRusherDefMeanNew = ('euclidian_opponent_1sec',np.nanmean))
        min_def_dist_new    = self.train_df[['PlayId', 'euclidian_opponent_1sec']].groupby('PlayId').agg(DistToRusherDefMinNew = ('euclidian_opponent_1sec',np.nanmin))

        mean_def_dist_half_new   = self.train_df[['PlayId', 'euclidian_opponent_halfsec']].groupby('PlayId').agg(DistToRusherDefMeanHalfNew = ('euclidian_opponent_halfsec',np.nanmean))
        min_def_dist_half_new    = self.train_df[['PlayId', 'euclidian_opponent_halfsec']].groupby('PlayId').agg(DistToRusherDefMinHalfNew = ('euclidian_opponent_halfsec',np.nanmin))

        self.play_train_df = self.play_train_df.merge(mean_def_dist, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(min_def_dist, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(mean_def_dist_new, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(min_def_dist_new, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(mean_def_dist_half_new, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(min_def_dist_half_new, how = 'left', on = 'PlayId')

        self.train_df.loc[self.train_df.euclidian_opponent.isnull(),'euclidian_opponent'] = 0
        self.train_df.loc[self.train_df.euclidian_home_1sec.isnull(),'euclidian_home_1sec'] = 0
        self.train_df.loc[self.train_df.euclidian_opponent_halfsec.isnull(),'euclidian_opponent_halfsec']= 0
        del mean_def_dist,min_def_dist,mean_def_dist_new,min_def_dist_new,mean_def_dist_half_new,min_def_dist_half_new
    
    
    def forward_box(self):
        
        def add_boxes(row, xcol, ycol): 
            
            if (row[ycol] >= 2.5) and (row[ycol] <= 7.5) and (row[xcol] <= 0) and (row[xcol] >= -5):
                return 1
            if (row[ycol] >= 2.5) and (row[ycol]<= 7.5) and (row[xcol] <= -5) and (row[xcol] >= -10):
                return 2
            if (row[ycol] >= 2.5) and (row[ycol] <= 7.5) and (row[xcol] <= -10) and (row[xcol] >= -15):
                return 3
            if ( abs(row[ycol]) <= 2.5) and (row[xcol] <= 0) and (row[xcol] >= -5):
                return 4
            if ( abs(row[ycol]) <= 2.5)  and (row[xcol] <= -5) and (row[xcol] >= -10):
                return 5
            if ( abs(row[ycol]) <= 2.5)  and (row[xcol] <= -10) and (row[xcol] >= -15):
                return 6
            if (row[ycol] <= -2.5) and (row[ycol] >= -7.5) and (row[xcol] <= 0) and (row[xcol]>= -5):
                return 7
            if (row[ycol] <= -2.5) and (row[ycol] >= -7.5) and (row[xcol] <= -5) and (row[xcol] >= -10):
                return 8
            if (row[ycol] <= -2.5) and (row[ycol] >= -7.5) and (row[xcol] <= -10) and (row[xcol] >= -15):
                return 9
            return np.NaN
        
        self.train_df['Box'] = self.train_df.apply(add_boxes, axis = 1,xcol = 'X_rel_rush', ycol = 'Y_rel_rush')
        self.train_df['BoxNew'] = self.train_df.apply(add_boxes, axis = 1,xcol = 'X_rel_rush_0.5sec_distance', ycol = 'Y_rel_rush_0.5sec_distance')
        self.train_df['BoxHalfNew'] = self.train_df.apply(add_boxes, axis = 1,xcol = 'X_rel_rush_1sec_distance', ycol = 'Y_rel_rush_1sec_distance')

        defs_in_b1 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.Box == 1),['PlayId','Box']].groupby('PlayId').count()
        defs_in_b2 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.Box == 2),['PlayId','Box']].groupby('PlayId').count()
        defs_in_b3 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.Box == 3),['PlayId','Box']].groupby('PlayId').count()
        defs_in_b4 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.Box == 4),['PlayId','Box']].groupby('PlayId').count()
        defs_in_b5 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.Box == 5),['PlayId','Box']].groupby('PlayId').count()
        defs_in_b6 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.Box == 6),['PlayId','Box']].groupby('PlayId').count()
        defs_in_b7 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.Box == 7),['PlayId','Box']].groupby('PlayId').count()
        defs_in_b8 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.Box == 8),['PlayId','Box']].groupby('PlayId').count()
        defs_in_b9 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.Box == 9),['PlayId','Box']].groupby('PlayId').count()

        defs_in_bcols = ['DefsInb1','DefsInb2','DefsInb3','DefsInb4','DefsInb5','DefsInb6','DefsInb7','DefsInb8','DefsInb9']



        defs_in_b1.columns = [defs_in_bcols[0]]
        defs_in_b2.columns = [defs_in_bcols[1]]
        defs_in_b3.columns = [defs_in_bcols[2]]
        defs_in_b4.columns = [defs_in_bcols[3]]
        defs_in_b5.columns = [defs_in_bcols[4]]
        defs_in_b6.columns = [defs_in_bcols[5]]
        defs_in_b7.columns = [defs_in_bcols[6]]
        defs_in_b8.columns = [defs_in_bcols[7]]
        defs_in_b9.columns = [defs_in_bcols[8]]



        self.play_train_df = self.play_train_df.merge(defs_in_b1, how = 'left', on = 'PlayId') 
        self.play_train_df = self.play_train_df.merge(defs_in_b2, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b3, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b4, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b5, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b6, how = 'left', on = 'PlayId') 
        self.play_train_df = self.play_train_df.merge(defs_in_b7, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b8, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b9, how = 'left', on = 'PlayId')


        #defenders in box HALF NEW
        defs_in_b1HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxHalfNew == 1),['PlayId','BoxHalfNew']].groupby('PlayId').count()
        defs_in_b2HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxHalfNew == 2),['PlayId','BoxHalfNew']].groupby('PlayId').count()
        defs_in_b3HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxHalfNew == 3),['PlayId','BoxHalfNew']].groupby('PlayId').count()
        defs_in_b4HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxHalfNew == 4),['PlayId','BoxHalfNew']].groupby('PlayId').count()
        defs_in_b5HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxHalfNew == 5),['PlayId','BoxHalfNew']].groupby('PlayId').count()
        defs_in_b6HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxHalfNew == 6),['PlayId','BoxHalfNew']].groupby('PlayId').count()
        defs_in_b7HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxHalfNew == 7),['PlayId','BoxHalfNew']].groupby('PlayId').count()
        defs_in_b8HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxHalfNew == 8),['PlayId','BoxHalfNew']].groupby('PlayId').count()
        defs_in_b9HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxHalfNew == 9),['PlayId','BoxHalfNew']].groupby('PlayId').count()

        defs_in_bcolsHN = ['DefsInb1HN','DefsInb2HN','DefsInb3HN','DefsInb4HN','DefsInb5HN','DefsInb6HN','DefsInb7HN','DefsInb8HN','DefsInb9HN']

        defs_in_b1HN.columns = [defs_in_bcolsHN[0]]
        defs_in_b2HN.columns = [defs_in_bcolsHN[1]]
        defs_in_b3HN.columns = [defs_in_bcolsHN[2]]
        defs_in_b4HN.columns = [defs_in_bcolsHN[3]]
        defs_in_b5HN.columns = [defs_in_bcolsHN[4]]
        defs_in_b6HN.columns = [defs_in_bcolsHN[5]]
        defs_in_b7HN.columns = [defs_in_bcolsHN[6]]
        defs_in_b8HN.columns = [defs_in_bcolsHN[7]]
        defs_in_b9HN.columns = [defs_in_bcolsHN[8]]

        self.play_train_df = self.play_train_df.merge(defs_in_b1HN, how = 'left', on = 'PlayId') 
        self.play_train_df = self.play_train_df.merge(defs_in_b2HN, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b3HN, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b4HN, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b5HN, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b6HN, how = 'left', on = 'PlayId') 
        self.play_train_df = self.play_train_df.merge(defs_in_b7HN, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b8HN, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b9HN, how = 'left', on = 'PlayId')

        #defenders in box NEW
        defs_in_b1N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxNew == 1),['PlayId','BoxNew']].groupby('PlayId').count()
        defs_in_b2N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxNew == 2),['PlayId','BoxNew']].groupby('PlayId').count()
        defs_in_b3N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxNew == 3),['PlayId','BoxNew']].groupby('PlayId').count()
        defs_in_b4N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxNew == 4),['PlayId','BoxNew']].groupby('PlayId').count()
        defs_in_b5N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxNew == 5),['PlayId','BoxNew']].groupby('PlayId').count()
        defs_in_b6N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxNew == 6),['PlayId','BoxNew']].groupby('PlayId').count()
        defs_in_b7N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxNew == 7),['PlayId','BoxNew']].groupby('PlayId').count()
        defs_in_b8N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxNew == 8),['PlayId','BoxNew']].groupby('PlayId').count()
        defs_in_b9N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.BoxNew == 9),['PlayId','BoxNew']].groupby('PlayId').count()

        defs_in_bcolsN = ['DefsInb1N','DefsInb2N','DefsInb3N','DefsInb4N','DefsInb5N','DefsInb6N','DefsInb7N','DefsInb8N','DefsInb9N']

        defs_in_b1N.columns = [defs_in_bcolsN[0]]
        defs_in_b2N.columns = [defs_in_bcolsN[1]]
        defs_in_b3N.columns = [defs_in_bcolsN[2]]
        defs_in_b4N.columns = [defs_in_bcolsN[3]]
        defs_in_b5N.columns = [defs_in_bcolsN[4]]
        defs_in_b6N.columns = [defs_in_bcolsN[5]]
        defs_in_b7N.columns = [defs_in_bcolsN[6]]
        defs_in_b8N.columns = [defs_in_bcolsN[7]]
        defs_in_b9N.columns = [defs_in_bcolsN[8]]

        self.play_train_df = self.play_train_df.merge(defs_in_b1N, how = 'left', on = 'PlayId') 
        self.play_train_df = self.play_train_df.merge(defs_in_b2N, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b3N, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b4N, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b5N, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b6N, how = 'left', on = 'PlayId') 
        self.play_train_df = self.play_train_df.merge(defs_in_b7N, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b8N, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_b9N, how = 'left', on = 'PlayId')    

        # how many defenders within distances
        defs_in_1 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent < 1),['PlayId','euclidian_opponent'] ].groupby('PlayId').count()
        defs_in_2 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent < 2),['PlayId','euclidian_opponent'] ].groupby('PlayId').count()
        defs_in_3 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent < 3),['PlayId','euclidian_opponent'] ].groupby('PlayId').count()
        defs_in_4 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent < 4),['PlayId','euclidian_opponent'] ].groupby('PlayId').count()
        defs_in_5 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent < 5),['PlayId','euclidian_opponent'] ].groupby('PlayId').count()
        defs_in_6 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent < 6),['PlayId','euclidian_opponent'] ].groupby('PlayId').count()
        defs_in_7 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent < 7),['PlayId','euclidian_opponent'] ].groupby('PlayId').count()
        defs_in_8 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent < 8),['PlayId','euclidian_opponent'] ].groupby('PlayId').count()
        defs_in_9 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent < 9),['PlayId','euclidian_opponent'] ].groupby('PlayId').count()
        defs_in_10 = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent < 10),['PlayId','euclidian_opponent'] ].groupby('PlayId').count()

        defs_in_cols = ['DefsIn1','DefsIn2','DefsIn3','DefsIn4','DefsIn5','DefsIn6','DefsIn7','DefsIn8','DefsIn9','DefsIn10']

        defs_in_1.columns = [defs_in_cols[0]]
        defs_in_2.columns = [defs_in_cols[1]]
        defs_in_3.columns = [defs_in_cols[2]]
        defs_in_4.columns = [defs_in_cols[3]]
        defs_in_5.columns = [defs_in_cols[4]]
        defs_in_6.columns = [defs_in_cols[5]]
        defs_in_7.columns = [defs_in_cols[6]]
        defs_in_8.columns = [defs_in_cols[7]]
        defs_in_9.columns = [defs_in_cols[8]]
        defs_in_10.columns = [defs_in_cols[9]]

        self.play_train_df = self.play_train_df.merge(defs_in_1, how = 'left', on = 'PlayId') 
        self.play_train_df = self.play_train_df.merge(defs_in_2, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_3, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_4, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_5, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_6, how = 'left', on = 'PlayId') 
        self.play_train_df = self.play_train_df.merge(defs_in_7, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_8, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_9, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_10, how = 'left', on = 'PlayId')


        # how many defenders within distances HALF NEW euclidian_opponent_halfsec
        defs_in_1HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_1sec < 1),['PlayId','euclidian_opponent_1sec'] ].groupby('PlayId').count()
        defs_in_2HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_1sec < 2),['PlayId','euclidian_opponent_1sec'] ].groupby('PlayId').count()
        defs_in_3HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_1sec < 3),['PlayId','euclidian_opponent_1sec'] ].groupby('PlayId').count()
        defs_in_4HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_1sec < 4),['PlayId','euclidian_opponent_1sec'] ].groupby('PlayId').count()
        defs_in_5HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_1sec < 5),['PlayId','euclidian_opponent_1sec'] ].groupby('PlayId').count()
        defs_in_6HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_1sec < 6),['PlayId','euclidian_opponent_1sec'] ].groupby('PlayId').count()
        defs_in_7HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_1sec < 7),['PlayId','euclidian_opponent_1sec'] ].groupby('PlayId').count()
        defs_in_8HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_1sec < 8),['PlayId','euclidian_opponent_1sec'] ].groupby('PlayId').count()
        defs_in_9HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_1sec < 9),['PlayId','euclidian_opponent_1sec'] ].groupby('PlayId').count()
        defs_in_10HN = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_1sec < 10),['PlayId','euclidian_opponent_1sec'] ].groupby('PlayId').count()

        defs_in_colsHN = ['DefsIn1HN','DefsIn2HN','DefsIn3HN','DefsIn4HN','DefsIn5HN','DefsIn6HN','DefsIn7HN','DefsIn8HN','DefsIn9HN','DefsIn10HN']

        defs_in_1HN.columns = [defs_in_colsHN[0]]
        defs_in_2HN.columns = [defs_in_colsHN[1]]
        defs_in_3HN.columns = [defs_in_colsHN[2]]
        defs_in_4HN.columns = [defs_in_colsHN[3]]
        defs_in_5HN.columns = [defs_in_colsHN[4]]
        defs_in_6HN.columns = [defs_in_colsHN[5]]
        defs_in_7HN.columns = [defs_in_colsHN[6]]
        defs_in_8HN.columns = [defs_in_colsHN[7]]
        defs_in_9HN.columns = [defs_in_colsHN[8]]
        defs_in_10HN.columns = [defs_in_colsHN[9]]

        self.play_train_df = self.play_train_df.merge(defs_in_1HN, how = 'left', on = 'PlayId') 
        self.play_train_df = self.play_train_df.merge(defs_in_2HN, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_3HN, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_4HN, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_5HN, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_6HN, how = 'left', on = 'PlayId') 
        self.play_train_df = self.play_train_df.merge(defs_in_7HN, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_8HN, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_9HN, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_10HN, how = 'left', on = 'PlayId')


        # how many defenders within distances NEW
        defs_in_1N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_halfsec < 1),['PlayId','euclidian_opponent_halfsec'] ].groupby('PlayId').count()
        defs_in_2N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_halfsec < 2),['PlayId','euclidian_opponent_halfsec'] ].groupby('PlayId').count()
        defs_in_3N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_halfsec < 3),['PlayId','euclidian_opponent_halfsec'] ].groupby('PlayId').count()
        defs_in_4N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_halfsec < 4),['PlayId','euclidian_opponent_halfsec'] ].groupby('PlayId').count()
        defs_in_5N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_halfsec < 5),['PlayId','euclidian_opponent_halfsec'] ].groupby('PlayId').count()
        defs_in_6N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_halfsec < 6),['PlayId','euclidian_opponent_halfsec'] ].groupby('PlayId').count()
        defs_in_7N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_halfsec < 7),['PlayId','euclidian_opponent_halfsec'] ].groupby('PlayId').count()
        defs_in_8N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_halfsec < 8),['PlayId','euclidian_opponent_halfsec'] ].groupby('PlayId').count()
        defs_in_9N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_halfsec < 9),['PlayId','euclidian_opponent_halfsec'] ].groupby('PlayId').count()
        defs_in_10N = self.train_df.loc[(~self.train_df.IsOnOffense) & (self.train_df.euclidian_opponent_halfsec < 10),['PlayId','euclidian_opponent_halfsec'] ].groupby('PlayId').count()

        defs_in_colsN = ['DefsIn1N','DefsIn2N','DefsIn3N','DefsIn4N','DefsIn5N','DefsIn6N','DefsIn7N','DefsIn8N','DefsIn9N','DefsIn10N']

        defs_in_1N.columns = [defs_in_colsN[0]]
        defs_in_2N.columns = [defs_in_colsN[1]]
        defs_in_3N.columns = [defs_in_colsN[2]]
        defs_in_4N.columns = [defs_in_colsN[3]]
        defs_in_5N.columns = [defs_in_colsN[4]]
        defs_in_6N.columns = [defs_in_colsN[5]]
        defs_in_7N.columns = [defs_in_colsN[6]]
        defs_in_8N.columns = [defs_in_colsN[7]]
        defs_in_9N.columns = [defs_in_colsN[8]]
        defs_in_10N.columns = [defs_in_colsN[9]]

        self.play_train_df = self.play_train_df.merge(defs_in_1N, how = 'left', on = 'PlayId') 
        self.play_train_df = self.play_train_df.merge(defs_in_2N, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_3N, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_4N, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_5N, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_6N, how = 'left', on = 'PlayId') 
        self.play_train_df = self.play_train_df.merge(defs_in_7N, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_8N, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_9N, how = 'left', on = 'PlayId')
        self.play_train_df = self.play_train_df.merge(defs_in_10N, how = 'left', on = 'PlayId')    

        all_count_cols = defs_in_cols + defs_in_colsN + defs_in_bcols + defs_in_bcolsN + defs_in_bcolsHN + defs_in_colsHN
        for col in all_count_cols:
            self.play_train_df.loc[self.play_train_df[col].isnull(),col] = 0.0