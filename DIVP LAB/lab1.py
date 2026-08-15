from PIL import Image, ImageEnhance

img = Image.open("peppers.png")

while True:
    print("\n===== IMAGE PROCESSING LAB =====")
    print("1. Display Image")
    print("2. Image Properties")
    print("3. Convert to Grayscale")
    print("4. Increase Brightness")
    print("5. Increase Contrast")
    print("6. Resize Image")
    print("7. Rotate Image")
    print("8. Flip Image")
    print("9. Crop Image")
    print("10. Negative Image")
    print("0. Exit")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        img.show()

    elif choice == 2:
        print("Size:", img.size)
        print("Format:", img.format)
        print("Mode:", img.mode)

    elif choice == 3:
        gray = img.convert("L")
        gray.show()

    elif choice == 4:
        bright = ImageEnhance.Brightness(img).enhance(2)
        bright.show()

    elif choice == 5:
        contrast = ImageEnhance.Contrast(img).enhance(2)
        contrast.show()

    elif choice == 6:
        resize = img.resize((300, 300))
        resize.show()

    elif choice == 7:
        rotate = img.rotate(90)
        rotate.show()

    elif choice == 8:
        flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip.show()

    elif choice == 9:
        crop = img.crop((50, 50, 250, 250))
        crop.show()

    elif choice == 10:
        negative = img.point(lambda p: 255 - p)
        negative.show()

    elif choice == 0:
        print("Thank You!")
        break

    else:
        print("Invalid Choice!")