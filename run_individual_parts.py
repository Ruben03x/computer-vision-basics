from computer_vision_assignment import ComputerVisionAssignment

def run_individual_parts():
    assignment = ComputerVisionAssignment()
    
    while True:
        print("\nSelect a part to run:")
        print("1. Color Thresholding (Greenscreen Removal)")
        print("2. Median Filter Denoising")
        print("3. Unsharp Masking")
        print("4. Bilinear Interpolation")
        print("5. Feature Detection and Matching")
        print("6. Scale Robustness Evaluation")
        print("7. Run All Parts")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-7): ").strip()
        
        try:
            if choice == "1":
                assignment.color_thresholding_greenscreen("greenscreen.jpg", "background.jpg")
                
            elif choice == "2":
                densities = [0.1, 0.2, 0.3]
                filter_sizes = [3, 5, 7]
                print("Choose desired density or else it will use 0.1, 0.2, 0.3")
                choice_density = input("Enter densities separated by commas (default: 0.1, 0.2, 0.3): ")
                if choice_density:
                    densities = [float(d) for d in choice_density.split(",")]
                assignment.median_filter_denoising("semper1.jpg", densities, filter_sizes)
                
            elif choice == "3":
                assignment.unsharp_masking("fox.jpg")
                
            elif choice == "4":
                scales = [0.5, 1.5, 2.0]
                assignment.image_resizing_comparison("semper1.jpg", scales)
                
            elif choice == "5":
                assignment.detect_and_match_features("semper1.jpg", "semper2.jpg")
                
            elif choice == "6":
                assignment.evaluate_scale_robustness("semper1.jpg")
                
            elif choice == "7":
                assignment.color_thresholding_greenscreen("greenscreen.jpg", "background.jpg")
                assignment.median_filter_denoising("semper1.jpg")
                assignment.unsharp_masking("fox.jpg")
                assignment.image_resizing_comparison("semper1.jpg")
                assignment.detect_and_match_features("semper1.jpg", "semper2.jpg")
                assignment.evaluate_scale_robustness("semper1.jpg")

            elif choice == "0":
                print("Terminated")
                break
                
            else:
                print("Invalid choice. Please enter 0-7.")
                
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print("Please check your inputs and try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    run_individual_parts()
