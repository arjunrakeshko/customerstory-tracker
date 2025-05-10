from database import CustomerStoryDB
import json
from datetime import datetime
from tabulate import tabulate
from typing import Dict, Any

def format_story(story: Dict[str, Any]) -> Dict[str, Any]:
    """Format a story for readable output."""
    return {
        'ID': story['case_study_id'],
        'Company': story['customer_name'],
        'Date Added': story['publication_date'],
        'Use Case': story['use_case'],
        'Location': story['customer_location'],
        'Insight Score': f"{story['insight_score']:.2f}"
    }

def main():
    """Main function to query and display stories."""
    try:
        # Initialize database connection
        db = CustomerStoryDB('customer_stories.db')
        
        # Get all stories
        stories = db.get_all_case_studies()
        if not stories:
            print("No stories found in the database.")
            return
        
        # Group stories by company
        stories_by_company = {}
        for story in stories:
            company = story['customer_name']
            if company not in stories_by_company:
                stories_by_company[company] = []
            stories_by_company[company].append(story)
        
        # Print summary
        print(f"\nTotal Stories: {len(stories)}")
        print(f"Companies: {len(stories_by_company)}")
        
        # Print stories by company
        print("\nStories by Company:")
        for company, company_stories in stories_by_company.items():
            print(f"\n{company} ({len(company_stories)} stories):")
            
            # Create table data
            table_data = [format_story(story) for story in company_stories]
            
            # Print table
            print(tabulate(
                table_data,
                headers='keys',
                tablefmt='grid',
                showindex=False
            ))
            
            # Print full details for each story
            for story in company_stories:
                print(f"\nStory ID: {story['case_study_id']}")
                print(f"Title: {story['title']}")
                print(f"URL: {story['url']}")
                print(f"Publication Date: {story['publication_date']}")
                print(f"Customer: {story['customer_name']}")
                print(f"Location: {story['customer_location']}")
                print(f"Industry: {story['customer_industry']}")
                print(f"Persona: {story['persona_title']}")
                print(f"Use Case: {story['use_case']}")
                print(f"Insight Score: {story['insight_score']:.2f}")
                
                if story['benefits']:
                    print("\nBenefits:")
                    for benefit in story['benefits']:
                        print(f"- {benefit}")
                
                if story['technologies']:
                    print("\nTechnologies:")
                    for tech in story['technologies']:
                        print(f"- {tech}")
                
                if story['partners']:
                    print("\nPartners:")
                    for partner in story['partners']:
                        print(f"- {partner}")
                
                print("\n" + "="*80)
        
    except Exception as e:
        print(f"Error querying database: {str(e)}")
        return

if __name__ == '__main__':
    main() 