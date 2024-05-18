
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import ttk, scrolledtext
from ttkthemes import ThemedStyle

class RecipeSuggestionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recipe Suggestion App")

        # Apply themed style
        style = ThemedStyle(root)
        style.set_theme("clam")

        self.create_widgets()

        # Load and preprocess the dataset
        self.dataset = pd.read_csv('C:\\Users\\hp\\Downloads\\RECIPIEGENERATOR\\Cleaned_Indian_Food_Dataset.csv')
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.X = self.vectorizer.fit_transform(self.dataset['Ingredients'])

        # Build a nearest neighbors model
        self.model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
        self.model.fit(self.X)

    def create_widgets(self):
        # Entry for user input
        self.input_entry = ttk.Entry(self.root, width=50)
        self.input_entry.grid(row=0, column=0, padx=10, pady=10)

        # Button to suggest recipes
        self.suggest_button = ttk.Button(self.root, text="Suggest Recipes", command=self.suggest_recipes)
        self.suggest_button.grid(row=0, column=1, padx=10, pady=10)

        # Text area to display suggested recipes
        self.result_text = scrolledtext.ScrolledText(self.root, width=60, height=10, wrap=tk.WORD)
        self.result_text.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    def suggest_recipes(self):
        user_input = self.input_entry.get()
        suggested_dishes = self.suggest_dishes(user_input)

        # Display suggested dishes in the text area
        self.result_text.delete(1.0, tk.END)  # Clear previous content
        self.result_text.insert(tk.END, "Suggested Dishes:\n")
        for i, dish in enumerate(suggested_dishes, 1):
            self.result_text.insert(tk.END, f"{i}. {dish}\n", f"tag_{i}")

            # Add tag to enable clicking on the dish name
            self.result_text.tag_configure(f"tag_{i}", foreground="blue", underline=True)
            self.result_text.tag_bind(f"tag_{i}", "<Button-1>", lambda event, dish_name=dish: self.show_dish_details(dish_name))

    def suggest_dishes(self, user_input):
         user_input_vector = self.vectorizer.transform([user_input])
         _, indices = self.model.kneighbors(user_input_vector)
         suggestions = self.dataset['RecipeName'].iloc[indices[0]].tolist()
         return suggestions


    def show_dish_details(self, dish_name):
        # Find details of the selected dish
        details = self.dataset.loc[self.dataset['RecipeName'] == dish_name, ['Instructions', 'Ingredients', 'Ingredient-count']].iloc[0]
        weights_str = str(details['Ingredient-count']) if pd.notna(details['Ingredient-count']) else ''
        instructions = details['Instructions']
        ingredients = details['Ingredients']

        # Create a new window to display details
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Details of {dish_name}")

        # Apply themed style to the details window
        style = ThemedStyle(details_window)
        style.set_theme("clam")

        # Label to display dish name
        dish_label = ttk.Label(details_window, text=f"Selected Dish: {dish_name}", font=("Helvetica", 14, "bold"))
        dish_label.pack(padx=10, pady=10)

        # Label to display instructions
        instructions_label = ttk.Label(details_window, text="Recipe Instructions:", font=("Helvetica", 12))
        instructions_label.pack(padx=10, pady=5)
        instructions_text = tk.Text(details_window, wrap=tk.WORD, width=60, height=10, font=("Helvetica", 12))
        instructions_text.insert(tk.END, instructions)
        instructions_text.pack(padx=10, pady=10)

        # Label to display ingredients
        ingredients_label = ttk.Label(details_window, text="Ingredients:", font=("Helvetica", 12))
        ingredients_label.pack(padx=10, pady=5)
        ingredients_text = tk.Text(details_window, wrap=tk.WORD, width=60, height=5, font=("Helvetica", 12))
        ingredients_text.insert(tk.END, ingredients)
        ingredients_text.pack(padx=10, pady=10)

        # Label to display ingredient weights
        weights_label = ttk.Label(details_window, text=f"Ingredient Weights: {weights_str}", font=("Helvetica", 12))
        weights_label.pack(padx=10, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = RecipeSuggestionApp(root)
    root.mainloop()
