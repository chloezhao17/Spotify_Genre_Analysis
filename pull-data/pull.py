#!/usr/bin/env python
# coding: utf-8

from config import secret, client
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from pprint import pprint

#create connection
client_credentials_manager = SpotifyClientCredentials(
    client_id=client, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# # gives the list of categories to pick from
# categories=[]
# for i in test["categories"]["items"]:
#     categories.append(i["id"])
# print(categories)


#list of categories that have been selected
categories=['hiphop', 'pop', 'country','rnb','latin', 'rock', 'edm_dance', 'indie_alt', 'classical', 'jazz', 'soul', 'kpop', 'reggae', 'blues']
playlist_list=[]


# pull the playlists
for cat in categories:
    try:
        category=sp.category_playlists(category_id=cat, country=None, limit=20, offset=0)
        for playlist in category["playlists"]["items"]:
            pprint(playlist)

            playlist_list.append({
                "name":playlist["name"],
                "pid":playlist["id"],
                "category":cat
            })
    except:
        continue
        
pprint(playlist_list)


len(playlist_list)


playlist=sp.playlist(playlist_list[0]["pid"])


pprint(playlist["tracks"]["items"][0]["track"])


# pull the sonngs in each playlist
songs=[]
for playlist in playlist_list:
    try:
        pl=sp.playlist(playlist["pid"])
        for song in pl["tracks"]["items"]:
            songs.append({
                "song_title":song["track"]["name"],
                "song_id":song["track"]["id"],
                "song_artist":song["track"]["artists"][0]["name"],
                "genre":playlist["category"],
                "playlist_id":playlist["pid"],
                "playlist_title":playlist["name"]  
            })
    except:
        continue
    


pprint(songs)


# See how much music we have for each category
song_df=pd.DataFrame(songs)
song_df.groupby(["genre"]).count()


# creater the music feature table and pull the data to append to it
to_df=pd.DataFrame(columns=['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms',
       'time_signature'])
for index, row in song_df.iterrows():
    song_features=sp.audio_features(tracks=row["song_id"])
    df=pd.DataFrame(song_features[0],index=[0])
    to_df=to_df.append(df)    


# Merge the above data into the song table
merge_df=song_df.merge(to_df, left_on="song_id", right_on="id")
merge_df
# song_df.loc[index,"danceability"]=song_features[0]["danceability"]


# drop dupicate songs
merge_df=merge_df.drop_duplicates("song_id")


merge_df


# export to csv
merge_df.to_csv("songs_with_features.csv")




