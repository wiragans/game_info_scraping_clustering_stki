#!/usr/bin/env python
# coding: utf-8

# In[1]:


## UAS SISTEM TEMU KEMBALI INFORMASI (STKI)

## ANGGOTA  : Wira Dwi Susanto (17.01.53.0053)
##            Sativa Wahyu Priyanto (17.01.53.0052)
##            Berliana Siwi Humandari (17.01.53.0103)
## Kelas    : B1


# In[2]:


pip install bs4


# In[3]:


pip install seaborn


# In[4]:


pip install wordcloud


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
get_ipython().run_line_magic('matplotlib', 'inline')

import requests
from bs4 import BeautifulSoup ##LOAD LIBRARY BEAUTIFULSOUP
import urllib.parse as urlparse
from urllib.parse import parse_qs


# In[6]:


action = "scraping" ## "scraping" atau "clustering"
game_id_start_scraping = 4614 ## Posisi Awal Game ID di Game Debate
game_id_end_scraping = 4914 ## Posisi Ending, Jangan terlalu banyak biar tidak lama scrapingnya
array_data_game_scraping = []

if action == "scraping":
    for game_id_start_scraping in range(int(game_id_start_scraping), int(game_id_end_scraping)+1):
        url = "https://www.game-debate.com/games/index.php?g_id=" + str(game_id_start_scraping)
        page = requests.get(url)
        #print(page.content) ##UNCOMMENT JIKA INGIN MELIHAT HASIL PRINT OUT YA
        
        parsed = urlparse.urlparse(url)
        game_id = str(parse_qs(parsed.query)['g_id'][0])
        game_id_int = int(game_id)
        #print("Game ID: " + game_id) ##UNCOMMENT JIKA INGIN MELIHAT HASIL PRINT OUT YA
        
        #GET RESOURCE HTML
        get_page = page.content
        
        #PARSE HTML DENGAN BEAUTIFULSOUP
        scraping_module = BeautifulSoup(get_page, 'html.parser')
        #print(scraping_module) ##UNCOMMENT JIKA INGIN MELIHAT HASIL PRINT OUT YA
        
        #Extract Page Title
        #game_title = scraping_module.find('div', attrs={"id":"gd-widget-div"})
        game_title = scraping_module.find('span', attrs={"itemprop":"name"})
        
        if game_title is None:
            game_title = "-"
        else:
            game_title = game_title.text
        
        game_title_replace = game_title.replace(" System Requirements ", "").strip()
        #print("Game Title: " + game_title_replace) ##UNCOMMENT JIKA INGIN MELIHAT HASIL PRINT OUT YA
        
        #Extract Game System Requirements
        game_spec_req = scraping_module.find('ul', attrs={"class":"devDefSysReqList"})
        
        if game_spec_req is None:
            game_spec_req = "-"
        else:
            game_spec_req = game_spec_req.text
        
        #print("System Requirements: \r\n" + game_spec_req) ##UNCOMMENT JIKA INGIN MELIHAT HASIL PRINT OUT YA
        
        #Extract Game Rating Based On User Review
        game_rating = scraping_module.find('span', attrs={"itemprop":"reviewRating"})
        
        if game_rating is None:
            game_rating = "0"
        else:
            game_rating = game_rating.text
            
        #print("Game Rating: " + game_rating + " / " + "10")
        #print("Game Ratingnya: " + game_rating) ##UNCOMMENT JIKA INGIN MELIHAT HASIL PRINT OUT YA
        
        #Extract Game Vote Based On User
        game_votes = scraping_module.find('div', attrs={"data-game-title":game_title_replace})
        
        if game_votes is None:
            game_votes = "0"
        else:
            game_votes = game_votes.get('data-rating-votes')
        
        #print("Total Game Votes: " + game_votes) ##UNCOMMENT JIKA INGIN MELIHAT HASIL PRINT OUT YA
        
        #Extract FPS System Benchmark Pada Game
        game_fps_bench = scraping_module.find('div', attrs={"class":"gamFpsValFigure"})
        
        if game_fps_bench is None:
            game_fps_bench = "0"
        else:
            game_fps_bench = game_fps_bench.text.strip()
        
        #print("FPS System Benchmark:\r\n" + game_fps_bench) ##UNCOMMENT JIKA INGIN MELIHAT HASIL PRINT OUT YA
        
        #Extract Game Short Desc
        game_short_desc = scraping_module.find('div', class_='game-description-container')
        
        if game_short_desc is None:
            game_short_desc = "-"
        else:
            game_short_desc = game_short_desc.text.strip()
        
        #print("Game Short Desc:\r\n" + game_short_desc) ##UNCOMMENT JIKA INGIN MELIHAT HASIL PRINT OUT YA
        
        array_data_game_scraping.append([int(game_id_int), str(game_title_replace), str(game_spec_req), float(game_rating), int(game_votes), str(game_fps_bench), str(game_short_desc)])
        
    print(array_data_game_scraping)


# In[7]:


##ADD DATA TO CSV UNTUK NANTINYA DILAKUKAN CLUSTERING
##KITA BUTUH LIBRARY PANDAS UNTUK PROSES PENGOLAHAN DATAFRAME-NYA
game_data = pd.DataFrame(array_data_game_scraping, columns=['game_id', 'game_title', 'system_requirements', 'game_rating', 'game_votes', 'fps_system_bench', 'game_short_desc'])
game_data.to_csv('game_data_set_new.csv', index=False)
print(game_data)


# In[8]:


## PROSES CLUSTERING, AMBIL DATASET PADA CSV DAN 5 DATA TERATAS
if action == "clustering": ## JIKA ACTION DIISI DENGAN "clustering" pada coding bagian paling atas
    game_data = pd.read_csv("game_data_set_new.csv")
    print(game_data.head())


# In[9]:


## Coba menampilkan informasi datatype pada dataset
game_data.info()


# In[10]:


game_data['length'] = game_data['game_title'].apply(len)
game_data.head()


# In[11]:


plt.rcParams['figure.figsize'] = (15, 7)
sns.distplot(game_data['length'], color = 'purple')
plt.title('The Distribution of Length over the Texts', fontsize = 20)


# In[12]:


## Gunakan fungsi wordcloud untuk visualisasi pada atribut game_title
wordcloud = WordCloud(background_color = 'lightcyan',
                      width = 1200,
                      height = 700).generate(str(game_data['game_title']))

plt.figure(figsize = (15, 10))
plt.imshow(wordcloud)
plt.title("WordCloud ", fontsize = 20)


# In[13]:


## Gunakan fungsi CountVectorizer untuk mencari word frequency
cv = CountVectorizer()
words = cv.fit_transform(game_data['game_title'])
sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

color = plt.cm.twilight(np.linspace(0, 1, 20))
frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = color)
plt.title("Most Frequently Occuring Words - Top 20")


# In[14]:


print("Shape of X :", words.shape)


# In[15]:


## Hapus kolom variable yang tidak diperlukan untuk proses clustering
game_data = game_data.drop(['game_id', 'system_requirements', 'fps_system_bench'], axis = 1)
game_data.head()


# In[16]:


## Menentukan variabel mana yang akan dicluster
## Misalnya variabel yang dipilih adalah game_rating, game_votes
game_data_x = game_data.iloc[:, 1:3]
game_data_x.head()


# In[17]:


## Lihat persebaran data tersebut dengan cara berikut ini, manfaatkan fungsi Library Seaborn pada Python
## dari grafik di bawah ini, terlihat bahwa data kebanyakan memiliki game rating 0.0 alias belum ada review
sns.scatterplot(x="game_rating", y="game_votes", data=game_data, s=50, color="red", alpha = 0.5)


# In[18]:


## Ubah variabel yang sebelumnya berbentuk dataframe menjadi sebuah array
x_array = np.array(game_data_x)
print(x_array)


# In[19]:


## Lakukan standarisasi
## Terlihat bahwa hasil dari scalling data membuat data yang kita miliki berada di antara 0 – 9
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
x_scaled


# In[20]:


## Tentukan jumlah clusternya, misalnya di sini ada 5 cluster
# Gunakan fungsi library K-Means pada Python
kmeans = KMeans(n_clusters = 5, random_state=123)
# Menentukan kluster dari data
kmeans.fit(x_scaled)


# In[21]:


## Cari nilai pusat dari masing masing cluster
print(kmeans.cluster_centers_)


# In[22]:


## Tampilkan hasil clustering dan tambahkan kolom clustering ke dalam dataframe
## Menampilkan hasil cluster
print(kmeans.labels_)

# Menambahkan kolom "cluster" dalam dataframe game_data_set info
game_data["cluster"] = kmeans.labels_
game_data.head()


# In[23]:


## Visualisasikan hasil clustering, kita manfaatkan fungsi library matplotlib pada Python
## Data Game Data Info telah berhasil di-cluster menjadi 5 cluster
fig, ax = plt.subplots()
sct = ax.scatter(x_scaled[:,1], x_scaled[:,0], s = 50,
c = game_data.cluster, marker = "o", alpha = 0.5)
centers = kmeans.cluster_centers_
ax.scatter(centers[:,1], centers[:,0], c='blue', s=100, alpha=0.5);plt.title("Hasil Clustering Menggunakan K-Means Untuk Dataset Game Info")
plt.xlabel("Scaled Game Rating")
plt.ylabel("Scaled Game Votes")
plt.show()


# In[24]:


## K-MEANS Top terms atau kata kunci per cluster
documents = game_data['game_title'] ## AMBIL DIDASARKAN PADA GAME_TITLE

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms atau kata kunci per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

keyword1 = "top simulator game"
Y = vectorizer.transform([keyword1])
prediction = model.predict(Y)
print(prediction)

keyword2 = "freestyle game"
Y = vectorizer.transform([keyword2])
prediction = model.predict(Y)
print(prediction)

keyword3 = "racing best game"
Y = vectorizer.transform([keyword3])
prediction = model.predict(Y)
print(prediction)

keyword4 = "most anticipated game"
Y = vectorizer.transform([keyword4])
prediction = model.predict(Y)
print(prediction)

keyword5 = "ghost recon"
Y = vectorizer.transform([keyword5])
prediction = model.predict(Y)
print(prediction)


# In[ ]:




