from collections import defaultdict
import clip
import numpy as np
import torch
from transformers import DistilBertTokenizer
from clip_vit.config import CFG

class kg_load:
    def __init__(self, args):
        self.args = args
    
    def load_kg(self):
        
        if (self.args.dataset == 'cifar' or self.args.dataset == 'flickr'):  
        
            classes = ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            
            kg_dict = defaultdict()
        
            kg_dict[0] = ['Wingspan: long or short', 'Fuselage shape: aerial', 'Jet engines']
            kg_dict[1] = ['Shape: sedan, or suv', 'color: red, blue', 'engine: small/ large' ]
            kg_dict[2] = ['Can fly', 'colorful', 'small bill']
            kg_dict[3] = ['Fluffy fur', 'sharp eyes', 'sharp teeth']
            kg_dict[4] = ['run fast', 'spotty skin', 'large eyes']
            kg_dict[5] = ['Dense fur', 'Sharp claws', 'run faster']
            kg_dict[6] = ['Webbed feet for swimming and jumping', 'Moist, smooth skin.', 'Bulging eyes on the sides of the head.']                         
            kg_dict[7] = ['Strong and muscular body', 'Long, flowing mane and tail.', 'Hooves for running and walking.']
            kg_dict[8] = ['Sturdy hull for navigating through water.', 'Multiple decks for accommodation and activities.', 'Navigational equipment such as radar and compass.']
            kg_dict[9] = ['Large cargo bed for transporting goods.', 'Robust suspension for handling heavy loads.', 'Powerful engine for towing and hauling.']
        
        elif (self.args.dataset == 'coco'):
            classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',  'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
            'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

           
            kg_dict = defaultdict()
            
            kg_dict['person'] = ['Human figure', 'Man or woman', 'Wears shirts, pants, coats', 'hairs like curly, straight, blonde, brunette', 'Smiling, crying, happy or sad in facial expression']
            kg_dict['bicycle'] = ['Two wheels', 'Handlebars', 'Pedals', 'Chain or gears', 'Tires']
            kg_dict['car'] = ['Four wheels', 'Engine', 'Windows', 'Headlights and taillights', 'Bumpers']
            kg_dict['motorcycle'] = ['Two wheels', 'Handlebars', 'Engine', 'Headlights amd taillights', 'Trottle']
            kg_dict['airplane'] = ['Wings', 'Tail', 'Cockpit', 'jet Engine', 'Fuselage']
            kg_dict['bus'] = ['Multi wheel', 'Long size', 'Large windows', 'Hardlights and taillights', 'Seats']
            kg_dict['train'] = ['Multiple carriages', 'Tracks', 'Front Engine/locomotive', 'Steel/Metal', 'Electrical lines']
            kg_dict['truck'] = ['Large size', 'Cargo capacity', 'Headlights and taillights', 'Huge tires', 'Big windshield']
            kg_dict['boat'] = ['Hull', 'Oars or Engine', 'Watercraft', 'Floating vessel', 'Bow or stern']
            kg_dict['traffic light'] = ['Signal lights', 'Pole', 'Intersection control', 'Red, green, yellow color', 'Electrical wires']
            kg_dict['fire hydrant'] = ['Red cylindar', 'Water outlet', 'firefighting', 'Roadside', 'Metal build']
            kg_dict['stop sign'] = ['Octagonal shape', 'Red color', 'White text', 'Traffic sign', 'Caution']
            kg_dict['parking meter'] = ['Tall post', 'Coin slot', 'Time display', 'Metalic structure', 'Signage']
            kg_dict['bench'] = ['Seating surface', 'Backrest', 'Outdoor furniture', 'Side tables', 'Park ammenity']
            kg_dict['bird'] = ['Feathers', 'Beak', 'Wings', 'Can fly', 'Sharp claws']
            kg_dict['cat'] = ['Sharp whiskers', 'Paw pads', 'Soft fur', 'Curved tail', 'Pointed ears']
            kg_dict['dog'] = ['Wagging tail', 'Snout', 'Wet nose', 'Expressive eyes', "Cannie eyes"]
            kg_dict['horse'] = ['Mane', 'Hooves', 'Muscular legs', 'Strong body', 'Four legs']
            kg_dict['sheep'] = ['Wooly coat', 'Hooves', 'Flock animal', 'Curved horns', 'Dropping ears']
            kg_dict['cow'] = ['Pointy horns', 'Udder', 'swivel ears', 'Thick hair', 'Four legs with hooves']
            kg_dict['elephant'] = ['Trunk', 'Tusks', 'Large size', 'Small tail', 'Big floppy ears']
            kg_dict['bear'] = ['Sharp claws', 'Thick fur', 'Strong and muscular body', 'killer animal', 'Prominent snout and nose' ]
            kg_dict['zebra'] = ['Black stripes', 'Horse-like body', 'Grassland dweller', 'Large round ears', 'Black and white mane']
            kg_dict['giraffe'] = ['Long neck', 'Spotted coat', 'Long legs hooves', 'Long eyelashes', 'Spotted coat']
            kg_dict['backpack'] = ['Shoulder straps', 'Compartments', 'Portable storage', 'Padded back', 'Zipper or buckle  ']
            kg_dict['umbrella'] = ['Canopy', 'Handle or grip', 'Rain protection', 'Metal frames', 'Foldable design']
            kg_dict['handbag'] = ['Handles or straps', 'Interior compartments', 'Fashion accessory', 'Zipper button', 'Pocket']
            kg_dict['tie'] = ['Neckwear', 'Knot', 'Formal attire', 'Long slender shape', 'Knot and loop']
            kg_dict['suitcase'] = ['Handle', 'Wheels', 'Luggage storage', 'Rectengular shape', 'Durable and sturdy']
            kg_dict['frisbee'] = ['Circular shape', 'Flying disc', 'Recreational toy', 'Disc shape', 'Curved edge']
            kg_dict['skis'] = ['Long and narrow', 'Bindings', 'Winter sports equipment', 'Carbon fiber', 'Used in icy surface']
            kg_dict['snowboard'] = ['Wide and flat', 'Bindings', 'Snow sports equipment', 'Gliding to snow', 'Pointy nose']
            kg_dict['sports ball'] = ['Spherical shape', 'Used in various sports', 'different colors', 'small in size', 'Elastic and bouncy']
            kg_dict['kite'] = ['Frame', 'Long tails', 'String attachment', 'Flies in sky', 'Lightweight']
            kg_dict['baseball bat'] = ['Long and cylindrical', 'Grip handle', 'Used in baseball', 'Made of wood', 'Long barrel']
            kg_dict['baseball glove'] = ['Handwear', 'Webbed pocket', 'Used in baseball', 'Latching and stiching', 'Swing and grip']
            kg_dict['skateboard'] = ['Flat board made of wood', 'Metal T-shaped components mounted underneath the deck', 'Four small wheels', 'Kicktail at back', 'Concave shape']
            kg_dict['surfboard'] = ['Long and narrow', 'Fin(s)', 'Used for surfing', "Curved nose", 'Wax traction pad at back']
            kg_dict['tennis racket'] = ['Oval frame', 'Strings streched tightly', 'Grip handle', 'String pattern spin generation', 'Swing capability']
            kg_dict['bottle'] = ['Container', 'Narrow neck', 'Cap or lid', 'Made of glass', 'Contains liquid']
            kg_dict['wine glass'] = ['Stem', 'Bowl', 'Used for wine tasting', 'Clear glass', 'Delicate and thin rim']
            kg_dict['cup'] = ['Handle', 'Cylindrical shape', 'Used for drinking', 'Made of glass', 'Contains liquid']
            kg_dict['fork'] = ['Prongs', 'Handle', 'Eating utensil', 'Pointy tines', 'Made of stainless steel']
            kg_dict['knife'] = ['Blade', 'Grip handle', 'Cutting tool', 'Made of stainless steel', 'Uses as cooking utensils']
            kg_dict['spoon'] = ['Bowl', 'Handle', 'Eating utensil', 'Used for lifting food', 'Scooping, stiring']
            kg_dict['bowl'] = ['Round shape', 'Used for serving food', 'Made of stainless steel', 'Hollow and open top', 'Lid']
            kg_dict['banana'] = ['Peel', 'Curved shape', 'Edible fruit', 'Yellow or green in color', 'White inside']
            kg_dict['apple'] = ['Round shape', 'Stem', 'Edible fruit', 'Red or green in color', 'White inside']
            kg_dict['sandwich'] = ['Layers of ingredients', 'Bread slices', 'Portable meal', 'Meat egg staffing', 'Triangular or oval shape']
            kg_dict['orange'] = ['Citrus fruit', 'Peel', 'Juicy and tangy', 'Sengemnted flesh inside', 'Orange color']
            kg_dict['broccoli'] = ['Green florets', 'Stalk', 'Nutritious vegetable', 'Green on color', 'Leafy green foliage']
            kg_dict['carrot'] = ['Orange color', 'Root vegetable', 'Tapered, cylindrical shape', 'Thin leafy green tops', 'Root texture']
            kg_dict['hot dog'] = ['Long bun', 'Sausage', 'Condiments', 'Light brown and golden color', 'Grill marks']
            kg_dict['pizza'] = ['Round shape', 'Toppings', 'Baked dish', 'Golden brown crust', 'Tomato sauce layer']
            kg_dict['donut'] = ['Ring shape', 'Glazed or filled', 'Sweet pastry', 'Hole in the center', 'Sprinkles or fillings']
            kg_dict['cake'] = ['Baked dessert', 'Layers', 'Frosting', 'Round or square shape', 'Different textures']
            kg_dict['chair'] = ['Seat', 'Backrest', 'Supportive furniture', 'Four legs', 'Made of wood, metal, plastic']
            kg_dict['couch'] = ['Padded seating', 'Armrests', 'Comfortable furniture', 'Cushion', 'Pillows']
            kg_dict['potted plant'] = ['Container', 'Growing medium', 'Indoor decoration', 'Spiky, wavy leaf', 'Lush green foliage']
            kg_dict['bed'] = ['Mattress', 'Bed frame', 'Sleeping furniture', 'Rectengular shape', 'Bedding pillows']
            kg_dict['dining table'] = ['Flat surface', 'Multiple wooden legs', 'Used for dining', 'Assorted table tops woods, glass, marble', 'Accompanied by chairs']
            kg_dict['toilet'] = ['Bowl', 'Seat', 'Flush mechanism', 'Porcelaine or ceramic made', 'Elongated or round shape']
            kg_dict['tv'] = ['Screen', 'Remote control', 'Entertainment device', 'Rectangular shape', 'Touch or button control']
            kg_dict['laptop'] = ['Portable computer', 'Screen', 'Keyboard', 'Touchpad or trackpad', 'Ports']
            kg_dict['mouse'] = ['Pointing device', 'Buttons', 'Cursor control', 'Scroll wheel', 'left or right clickable button']
            kg_dict['remote'] = ['Control device', 'Buttons', 'Wireless communication', 'Power or volume control', 'Rectengular shape']
            kg_dict['keyboard'] = ['Input device', 'Keys', 'Typing interface', 'Arrays of keys in a layout', 'Rectengular shape']
            kg_dict['cell phone'] = ['Mobile device', 'Touchscreen', 'Communication tool', 'Rectengular shape', 'Front and rear cameras']
            kg_dict['microwave'] = ['Rectangular box shape', 'Control panel with button', 'Turnable inside rotating base', 'Rectangular shape', 'Glass window for viewing']
            kg_dict['oven'] = ['Rectangular box shape', 'Baking chamber', 'Temperature control', "oven rack", 'Differnt mode with buttons']
            kg_dict['toaster'] = ['Rectangular  boxshape', 'Slots', 'Toasting bread', 'Lever for lowering and rising bread', 'Control buttons and knob']
            kg_dict['sink'] = ['Basin', 'Faucet', 'Water drainage', 'made of ceramic', 'Drainage systems and plumbing connections']
            kg_dict['refrigerator'] = ['Rectangular box shape', 'Cooling compartment', 'Food storage', 'Comapartments inside', 'Freezer inside']
            kg_dict['book'] = ['Pages', 'Thick or slim', 'Written content and image', 'Rectangualar shape', 'hardcover or paperback']
            kg_dict['clock'] = ['Time display', 'Hour and minute hands', 'Timekeeping device', 'Oval, round, or rectangular shape', 'Analog or digital']
            kg_dict['vase'] = ['Container', 'Decorative', 'Used for holding flowers', 'Narrow neck', 'Cylindrical, spherical, asymmetrical']
            kg_dict['scissors'] = ['Two blades', 'Handles', 'Cutting tool', 'Made of stainless steel', 'Sharp and pointy tip']
            kg_dict['teddy bear'] = ['Soft plush', 'Stuffed toy', 'Cuddly companion', 'Soft dense furry', 'Brown color']
            kg_dict['hair drier'] = ['Blow dryer', 'Handle', 'Hot air flow', 'Nozle for airflow', 'Pistol shape']
            kg_dict['toothbrush'] = ['Handle', 'Bristles', 'Oral hygiene tool', 'Small elongated shape', 'Cleaning tool']

            # kg_dict['person'] = ['Diverse individuals engaged in various activities, displaying unique physical features and clothing styles.', 'Exhibit a wide range of emotions and interactions, forming the essence of human society.']
            # kg_dict['bicycle'] = ['Two-wheeled transport with a frame, handlebars, pedals, and wheels, commonly used for commuting and recreational purposes.', 'Various styles and colors.']
            # kg_dict['car'] = ['Four-wheeled motor vehicle designed for personal transportation, featuring an enclosed passenger compartment, windows, and doors.', 'Cars come in different shapes, sizes, and colors.']
            # kg_dict['motorcycle'] = ['Motorized two-wheeled vehicle with a powerful engine, handlebars, and an open design.', 'Various styles and colors']
            # kg_dict['airplane'] = ['Flying machine with wings, engines, a fuselage, and windows, designed for air travel and transportation of passengers or cargo.', 'Come in different sizes and configurations, enabling long-distance travel at high speeds.']
            # kg_dict['bus'] = ['Large public transport vehicle equipped with multiple rows of seats, windows, and doors.', 'Buses play a vital role in urban and intercity transportation.']
            # kg_dict['train'] = ['Long vehicle that runs on tracks, consisting of locomotives and connected carriages, used for transporting passengers or goods.', 'Trains offer a comfortable and efficient means of travel']
            # kg_dict['truck'] = [' Heavy-duty vehicle designed for transporting goods and cargo, featuring a large cargo area at the back.', 'Trucks come in various sizes and configurations, serving as the backbone of logistics.']
            # kg_dict['boat'] = ['Watercraft used for navigation on water, available in different types such as sailboats, motorboats, or canoes.', 'Boats come in various sizes and shapes, serving purposes ranging from leisure and recreation to transportation and exploration.']
            # kg_dict['traffic light'] = ['Intersection signal consisting of red, yellow, and green lights, regulating the flow of vehicles and pedestrians.', 'Traffic lights ensure orderly and safe movement at road junctions.']
            # kg_dict['fire hydrant'] = ['Red or yellow firefighting equipment found in public areas, designed for easy access to water supply in case of emergencies.', 'Fire hydrants are typically cylindrical in shape.']
            # kg_dict['stop sign'] = ['Red octagonal traffic sign with large white lettering, indicating a mandatory stop at intersections.', 'Stop signs play a crucial role in regulating traffic flow and ensuring safety by providing clear instructions to drivers.']
            # kg_dict['parking meter'] = ['Devices used to collect fees for parking in designated areas, typically found in urban environments.', 'Parking meters are often tall and vertical in shape, featuring displays and coin slots to facilitate payment for parking.']
            # kg_dict['bench'] = ['Outdoor seating furniture, commonly made of wood or metal, providing a place to rest or relax.', 'Benches are characterized by their long, flat surface and legs, often found in parks, gardens, or public spaces.']
            # kg_dict['bird'] = ['Diverse avian creatures with a wide range of colors, sizes, and unique feather patterns.', 'Birds exhibit various behaviors such as flying, perching, and singing, contributing to the beauty of nature.']
            # kg_dict['cat'] = ['Small domesticated carnivorous mammals known for their agility and independence.', 'Cats come in different breeds, sizes, and colors, and they display characteristic behaviors such as purring and grooming.']
            # kg_dict['dog'] = ['Domesticated canines known for their loyalty and companionship to humans.', 'Dogs come in various breeds, sizes, and colors, exhibiting a wide range of behaviors and serving various roles, including working, service, and as pets.']
            # kg_dict['horse'] = ['Majestic and powerful animals known for their grace and speed.', 'Horses come in different breeds, sizes, and coat colors, playing significant roles in transportation, sports, and leisure activities.']
            # kg_dict['sheep'] = ['Domesticated ruminant mammals known for their wool and meat production.', 'Sheep typically have thick woolly coats and are often seen in flocks grazing in pastures.']
            # kg_dict['cow'] = ['Large domesticated ungulates, primarily raised for milk and meat production.', 'Cows are characterized by their robust bodies, distinctive horns, and characteristic "moo" sound.']
            # kg_dict['elephant'] = [' Enormous mammals with long trunks and ivory tusks, known for their intelligence and social behavior.', 'Elephants exhibit unique physical features, such as large ears and a massive body, and are symbols of strength and wisdom.']
            # kg_dict['bear'] = ['Powerful mammals found in various habitats, characterized by their strong bodies and sharp claws.', 'Bears come in different species, sizes, and colors, and they play significant roles in ecosystems.']
            # kg_dict['zebra'] = ['Striped equids known for their distinctive black and white patterns.', 'Zebras exhibit social behavior and can be found in grassland habitats, adding a touch of wild beauty to the African savannah.']
            # kg_dict['giraffe'] = ['Tallest living land animals with long necks and distinctive coat patterns.', 'Giraffes exhibit elegant movements and feed on leaves from tall trees, creating an iconic silhouette on the African plains.']
            # kg_dict['backpack'] = ['Portable bag carried on the back, typically made of durable materials, used for carrying belongings during travel, hiking, or daily activities.', 'Backpacks come in various sizes, designs, and colors, providing convenience and storage capacity.']
            # kg_dict['umbrella'] = ['Protective canopy device used for shielding against rain or sunlight.', 'Umbrellas consist of a collapsible frame and a waterproof or sun-blocking fabric, providing shelter and coverage.']
            # kg_dict['handbag'] = ['Small bag carried by hand or over the shoulder, designed to hold personal items such as wallets, keys, and cosmetics.', 'Handbags come in different styles, materials, and sizes, complementing fashion and functionality.']
            # kg_dict['tie'] = ['Neckwear accessory worn with shirts or suits.', 'Typically made of silk or other fabrics, adding a touch of sophistication and style to formal attire.']
            # kg_dict['suitcase'] = ['Travel bag designed for carrying clothes and personal belongings during trips.',  'Often equipped with wheels and a handle for easy transportation.']
            # kg_dict['frisbee'] = ['Disc-shaped flying toy made of plastic or other materials', 'Thrown through the air for recreational games and sports.']
            # kg_dict['skis'] = ['Narrow, elongated devices worn on the feet for gliding over snow', 'Used in various winter sports and activities.']
            # kg_dict['snowboard'] = ['Flat board-like device used for gliding over snow', 'Popular in snowboarding sports and recreational activities.']
            # kg_dict['sports ball'] = ['Spherical object used in various sports, such as soccer, basketball, or tennis.', 'Providing entertainment and facilitating gameplay.']
            # kg_dict['kite'] = ['Lightweight flying object with a framework covered in fabric', 'Flown in the wind for recreational purposes or kite-flying competitions.']
            # kg_dict['baseball bat'] = ['Cylindrical wooden or metal club', 'Used in baseball games to hit the ball thrown by the pitcher.']
            # kg_dict['baseball glove'] = ['Protective leather or synthetic mitt worn by players', 'To catch and field the ball.']
            # kg_dict['skateboard'] = ['Flat board made of wood', 'Metal T-shaped components mounted underneath the deck']
            # kg_dict['surfboard'] = ['Long and narrow', 'Wax traction pad at back']
            # kg_dict['tennis racket'] = ['Oval frame', 'Strings streched tightly']
            # kg_dict['bottle'] = ['Container typically made of glass, plastic, or metal, used for holding liquids such as water, beverages, or oils.', 'Narrow neck']
            # kg_dict['wine glass'] = ['Delicate glassware designed specifically for serving and enjoying wine.', 'Featuring a stem and a bowl-shaped cup.']
            # kg_dict['cup'] = ['Small open-top container without a handle, used for holding and drinking beverages like tea, coffee, or water.', 'Cylindrical shape']
            # kg_dict['fork'] = ['Utensil with two or more prongs used for lifting and eating food', 'Stainless steel made handle']
            # kg_dict['knife'] = ['Sharp-edged utensil used for cutting and slicing food.', 'A handle and a blade made of metal or other materials.']
            # kg_dict['spoon'] = ['Utensil with a shallow bowl and a handle.', 'Stirring, scooping, and consuming liquids or semi-solid foods.']
            # kg_dict['bowl'] = ['Deep, round-shaped dish used for serving', "Hollow top"]
            # kg_dict['banana'] = ['Curved shape', 'Elongated fruit with a yellow peel.', ]
            # kg_dict['apple'] = ['Round fruit with a smooth or textured skin', 'Typically red, green, or yellow']
            # kg_dict['sandwitch'] = ['Food item consisting of layers of bread.', 'Various ingredients like meat, cheese, vegetables, or spreads.']
            # kg_dict['orange'] = ['Citrus fruit with a bright orange-colored peel.', 'Known for its refreshing juice and sweet-tart flavor.']
            # kg_dict['broccoli'] = ['Nutritious vegetable with green florets and thick stalks.', 'Often consumed steamed, stir-fried, or in salads.']
            # kg_dict['carrot'] = ['Root vegetable with an orange color.', 'Providing a crunchy texture and sweet flavor.']
            # kg_dict['hot dog'] = ['Cooked sausage served in a long bun.', 'Condiments and toppings, a popular street food.']
            # kg_dict['pizza'] = ['Savory dish consisting of a round or square-shaped crust', 'Topped with sauce, cheese, and various toppings, baked to perfection.']
            # kg_dict['donut'] = ['Circular pastry with a sweet, fried dough base, often coated with sugar or glaze', 'A delightful treat for breakfast or dessert.']
            # kg_dict['cake'] = ['Celebratory dessert made from layers of moist and fluffy sponge.', 'Delectable creams or icings.']
            # kg_dict['chair'] = ['Furniture designed for sitting, offering comfort', 'Various styles, materials, and designs to complement any space.']
            # kg_dict['couch'] = ['Large, upholstered seating furniture designed for relaxation and comfort', 'Living rooms or lounges.']
            # kg_dict['potted plant'] = ['Living plant cultivated in a pot or container', 'The beauty of nature indoors and adding a touch of greenery to homes and spaces.']
            # kg_dict['bed'] = ['Furniture designed for sleeping and resting.', 'Offering comfort and relaxation, square shape']
            # kg_dict['dinning table'] = ['Furniture for meals and gatherings.', 'Central space for sharing food and conversation.']
            # kg_dict['toilet'] = ['Sanitary fixture used for human waste disposal and hygiene', 'Component of modern bathrooms and restroom facilities.']
            # kg_dict['tv'] = ['Electronic device with a screen', 'Displays audiovisual content, providing entertainment, news.']
            # kg_dict['laptop'] = ['Portable computer device designed for mobility and productivity.', 'Computing power, internet connectivity, and versatility for work, study, and entertainment.']
            # kg_dict['mouse'] = ['Pointing device used to navigate and interact with a computer screen', 'Buttons and a scroll wheel, enabling precise cursor control.']
            # kg_dict['remote'] = ['Handheld device used to control electronic devices such as televisions, audio systems, or streaming devices', 'Convenient operation from a distance.']
            # kg_dict['keyboard'] = ['Input device with a set of keys for typing and entering commands into a computer or other devices', 'Essential for text input and interaction.']
            # kg_dict['cell phone'] = [' Mobile communication device enabling voice calls, messaging, internet access', 'Providing constant connectivity and convenience.']
            # kg_dict['micro wave'] = ['Quickly heat and cook food', 'Offering a convenient and time-saving cooking method for various meals and snacks.']
            # kg_dict['oven'] = ['Cooking appliance used for baking, roasting, and heating food', 'A controlled environment for precise cooking and creating delicious dishes.']
            # kg_dict['toaster'] = ['Appliance used to toast bread slices', 'Offering a quick and efficient way to enjoy crispy and golden-brown toast for breakfast or snacks.']
            # kg_dict['sink'] = ['Basin with a faucet for water supply and drainage', 'Washing hands, dishes, or various cleaning tasks in kitchens, bathrooms, or utility rooms.']
            # kg_dict['refrigerator'] = ['Appliance designed for food storage and preservation', 'Compartments and cooling technology to keep food fresh and extend its shelf life.']
            # kg_dict['book'] = ['Written or printed work consisting of pages bound together', 'Offering knowledge, entertainment, and storytelling, allowing readers to explore diverse topics and worlds.']
            # kg_dict['clock'] = ['Timekeeping device with a mechanism or digital display', 'Track and display the current time, ensuring punctuality and scheduling.']
            # kg_dict['vase'] = ['Container, often made of glass or ceramic', 'Displaying flowers or decorative arrangements, adding beauty and elegance to any space.']
            # kg_dict['scissors'] = ['Cutting tool with two sharp blades hinged in the middle', 'Precise cutting of paper, fabric, or other materials, facilitating various crafting and household tasks.']
            # kg_dict['teddy bear'] = [' Soft and cuddly stuffed toy, often shaped like a bear,', 'Comfort and companionship for children and adults alike.']
            # kg_dict['hair dryer'] = ['Electric device used for drying and styling hair', 'A heating element and airflow settings to achieve desired hairstyles.']
            # kg_dict['toothbrush'] = ['Oral hygiene tool with bristles for cleaning teeth', 'Dental health and freshness through regular brushing.']



            
                    
        return classes, kg_dict
    
    def get_kg_emb (self, kg_dict, row=10, col=3):
        
        lst_tokens = []
        for kg in kg_dict.values():
            kg_tokens = clip.tokenize(kg)
            lst_tokens.append(kg_tokens)
        
        stacked_tensors = torch.stack(lst_tokens)
        return stacked_tensors
    
    def kgemb_bert (self, kg_dict, tokenizer, row = 10, col = 3):
        lst_tokens = {}
        lst_input_ids = []
        lst_attention_masks = []
        for kg in kg_dict.values():
            kg_tokens = tokenizer.batch_encode_plus(kg, max_length=CFG.max_length, padding='max_length', truncation=True, return_tensors='pt')
            lst_input_ids.append(kg_tokens['input_ids'])
            lst_attention_masks.append(kg_tokens['attention_mask'])

        
        stacked_tensors_ids = torch.stack(lst_input_ids)
        stacked_tensors_masks = torch.stack(lst_attention_masks)
        
        lst_tokens['input_ids'] = stacked_tensors_ids
        lst_tokens['attention_mask'] = stacked_tensors_masks
        print('shape: ', stacked_tensors_ids.shape, stacked_tensors_masks.shape)
        return lst_tokens

    