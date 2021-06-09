---
layout: post
title: Project Reflection
---

Over the course of the quarter, I alongside my group Kyle Fang, Adhvaith Vijay, and Britney Zhao created a [webapp](http://pic16b-dog-detector.herokuapp.com/) for users who are learning about dog breeds. In this reflection, the first four sections were written as a group, and I wrote the last two sections individually.

## 1. Overall, what did you achieve in your project?

We achieved a variety of goals in these projects. Namely, we created a model capable of predicting over 120 dog breeds based on an image. Based on 10 epochs of training we achieved validation accuracy in the range of 80%. On top of our dog breed classifier, we also created an interactive web page for users to find out what dog breeds most align with their interest. Using categories such as ease of maintenance, dog size, and trainability we are able to gauge which dog breed(s) are best suited for an individual.

## 2. What are two aspects of your project that you are especially proud of?

One feature of our project is that the model learns from incorrect breed predictions when the user submits a Google Form containing the picture and correct dog breed. This is one portion of our project that we are particularly proud of, as the model actively learns from data beyond the original dataset we trained our model on. When a user inputs the incorrectly identified photo alongside the correct dog breed, this data is added to a spreadsheet that our model can then train with and learn from. On the web application, this learning process automatically occurs once a day, so within 24 hours, our model will implement the correct prediction of the same photo. We thought having the model learn from its mistakes was an innovative addition to our project, and are especially proud of it for this reason.

We are also very proud of the dog breed recommendation portion of our project. On the “Find Your Perfect Dog!” page of our web application, users can input a variety of attributes they want in a dog, and our project then returns the top three matches based on a dataset we found online. We made this page as user-friendly as possible, with an efficient matching algorithm to ensure a fast result, as well as hyperlinks that users can click on and go to the American Kennel Club website containing more information about the matched breeds. Users can then easily learn more about these dog breeds and make an informed decision regarding which breed best suits their lifestyle.

## 3. What are two things you would suggest doing to further improve your project?

One suggestion to further improve our project is to include more dog breeds. Currently, the breed prediction model is trained on 121 breeds and the breed recommender includes 199 breeds. There are many more dog breeds that could be included in these two aspects of our project. Furthermore, our breed prediction model is trained on images of purebred dogs, and thus it does not perform as well on mixed breed dogs. If we could obtain or create a database of mixed breed dogs that included the list of breeds that each dog is, we could further train the model to predict the multiple breeds of a dog. This does pose many challenges since the amount of breed combinations is very great, however, it would allow our project to be more inclusive of dogs.

Another suggestion to improve our project would be to add a ranking system to the breed recommender. Currently, the user selects their preferences on 6 dog features, and all of these features are weighted equally when recommending the dog breeds. It is likely that of the 6 presented features, a user may care about some features more than others. Perhaps they want a dog with minimal shedding and maintenance, but don’t care about the size of the dog. Adding a ranking system for the features would provide recommendations that better match the preferences of the user.

## 4. How does what you achieved compare to what you set out to do in your proposal?

We completed more than what we included in our proposal! Our project proposal only indicated a model on a webapp that would input a picture and output the dog breed. We successfully did this and added more features.

Our model also includes an online learning feature. The model can take feedback from the user and improve itself in the next run. We also included sample images, so people who don’t have photos handy on their devices can still enjoy the webapp. We also included a dog recommender tab where the user can input his or her preferences, and the webapp will use KD Trees to predict the top 3 matches and display corresponding pictures and links.


## 5. What are three things you learned from the experience of completing your project?

Something I learned from completing our project is the importance of understanding the data you are working with. My focus in this project was creating the breed recommender. After finding a dataset that listed 199 dog breeds and gave a numeric value to each of the 12 features (size, intelligence, maintenance, etc.), my first instinct was to create a TensorFlow model like the ones we made for blog post 3. However, after discussing with team members, I realized that with the data format, it was as if each dog breed only had one data point associated with it. Thus, it would not be feasible to train and test a model as we had learned in class. Instead, we had to brainstorm new ways to give a breed recommendation to the user.

Another thing I learned from working on the breed recommender is the importance of evaluating your methods and getting feedback from others. The first iteration of the breed recommender used cosine similarity. After performing feature selection, we had 6 dog features that users could indicate their preferences about. We treated the input as a vector of length 6 and compared it to the 199 breeds in the dataset with cosine similarity. The angle between the input vector and the vector that each breed represents was computed, and then the breed with the smallest angle was recommended. Initially, we thought we had finished, but after testing it and discussing it with group members, I realized that cosine similarity compares the angle between vectors but not the magnitude. Thus any two inputs that are scalar multiples of each other would have the same recommendations. Clearly, someone who prefers a petite, low maintenance, quiet dog should not be given the same recommendation as someone who desires a large, high maintenance, loud dog. It was only through thorough testing and discussion with peers that I realized this flaw in the recommender. I  had to take a step back and find a way to compare both the magnitude and angle of the vectors to provide a recommendation.

Additionally, throughout this project, I learned new Python skills. I learned how to implement cosine similarity to give dog breed recommendations. Later, we implemented a KD tree data structure to give recommendations. Creating the KD tree structure allowed us to find the “nearest neighbor” in the tree to the input vector. This solved the issue that the first breed recommender had where it gave the same results when the input vectors were scalar multiples of each other. Through this project, I learned about multiple ways to give recommendations to users based on their preferences, and the strengths and weaknesses that these methods have.


## 6. How will your experience completing this project will help you in your future studies or career? Please be as specific as possible.

Completing this project gave me experience working with a team. With four people and a common goal, it was important for us to be communicating with each other. Additionally, we had to make sure our code was clear, both in style and also with comments and clear commit messages. Working on this project also gave me experience using git and version control. The experience completing this project will definitely help me in the future where I expect to be working on projects with coworkers that contain many moving parts. It will be crucial for me to be a clear communicator, solicit feedback from others, and write clear and documented code.
