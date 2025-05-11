from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
import pandas as pd

# Set up Selenium WebDriver
service = Service('D:/msedgedriver.exe')  # Replace with the correct path
edge_options = Options()
edge_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
driver = webdriver.Edge(service=service, options=edge_options)
driver.implicitly_wait(2)

# Function to extract post details
def extract_post_details(raw_content, driver, link):
    try:
        submission_id = re.search(r'/comments/([a-z0-9]+)/', link).group(1)
    except AttributeError:
        submission_id = "Submission ID not found"

    try:
        time_date = driver.find_element(By.TAG_NAME, "time").get_attribute("title")
    except Exception:
        time_date = "Time/Date not found"

    try:
        title = driver.find_element(By.TAG_NAME, "h1").text
    except Exception:
        title = "Title not found"

    lines = raw_content.split('\n')
    user_id = None
    filtered_content = []
    recording_content = False

    for i, line in enumerate(lines):
        if '•' in line and i + 2 < len(lines):
            user_id = lines[i + 2].strip()
            recording_content = True

        if recording_content:
            if "Upvote" in line:
                break
            if "•" not in line and "mo. ago" not in line and user_id not in line:
                filtered_content.append(line.strip())

    return title, user_id, time_date, '\n'.join(filtered_content), submission_id

# List of subreddits and search terms
subreddits = ["Philippines"]
search_terms = ["Technology"]
time_filter = "year"  # Use "month", "week", etc., for other ranges

# Collect data
data = []
MAX_RETRIES = 3  # Retry limit for loading posts
POST_LIMIT = 5   # Limit to save exactly 5 posts
visited_links = set()  # Track visited links

try:
    for subreddit in subreddits:
        for term in search_terms:
            print(f"Searching in subreddit: {subreddit}, for term: {term}, time filter: {time_filter}")
            
            # Construct the search URL with time filter
            search_url = (
                f"https://www.reddit.com/r/{subreddit}/search?"
                f"q={term}&restrict_sr=1&sort=new&t={time_filter}"
            )
            driver.get(search_url)
            time.sleep(5)

            last_height = driver.execute_script("return document.body.scrollHeight")

            while len(data) < POST_LIMIT:
                try:
                    posts = driver.find_elements(By.CSS_SELECTOR, "a[data-testid='post-title-text']")
                    for i in range(len(posts)):
                        try:
                            # Dynamically re-locate the post element
                            posts = driver.find_elements(By.CSS_SELECTOR, "a[data-testid='post-title-text']")
                            post = posts[i]
                            link = post.get_attribute("href")

                            if link and link not in visited_links:
                                visited_links.add(link)  # Add to visited links

                                retries = 0
                                while retries < MAX_RETRIES:
                                    try:
                                        driver.get(link)
                                        time.sleep(3)

                                        raw_content = WebDriverWait(driver, 15).until(
                                            EC.presence_of_element_located((By.TAG_NAME, "body"))
                                        ).text

                                        title, user_id, time_date, post_content, submission_id = extract_post_details(raw_content, driver, link)

                                        # Add data to the list
                                        data.append({
                                            "Title": title,
                                            "User ID": user_id,
                                            "Time/Date": time_date,
                                            "Content": post_content,
                                            "Submission ID": submission_id
                                        })
                                        print(f"\rCollected {len(data)}/{POST_LIMIT} posts...", end="")
                                        break
                                    except Exception as e:
                                        retries += 1
                                        if retries == MAX_RETRIES:
                                            print(f"\nError processing {link}: {e}")
                        except Exception as e:
                            print(f"Encountered issue with post element: {e}")
                            continue

                    # Scroll down if not enough posts found
                    if len(data) < POST_LIMIT:
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        time.sleep(3)
                        new_height = driver.execute_script("return document.body.scrollHeight")
                        if new_height == last_height:
                            break
                        last_height = new_height
                except Exception as main_exception:
                    print(f"Error while processing posts: {main_exception}")
                    break
finally:
    driver.quit()

# Save data to Excel
if not data:
    print("\nNo posts were collected.")
else:
    df = pd.DataFrame(data)
    df.to_excel(f"Reddit_Posts_Filtered_By_{time_filter}.xlsx", index=False, engine="openpyxl")
    print(f"\nSaved {len(data)} posts to 'Reddit_Posts_Filtered_By_{time_filter}.xlsx'.")
