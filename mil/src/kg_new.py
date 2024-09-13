from collections import defaultdict
import clip
import numpy as np
import torch
from transformers import DistilBertTokenizer

########
# Prompt: Give 5 visual properties of a [class] in the form a phrase, and then for each property, provide 3 addition sub-properties of that property in the form of a phrase. Each property should be between 1-4 words.
########

class kg_load:
    def __init__(self, args):
        self.args = args
    
    def load_kg(self):
        
        if (self.args.dataset == 'cifar' or self.args.dataset == 'flickr1'):  
        
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
        
        elif (self.args.dataset == 'coco' or self.args.dataset == 'flickr1' or self.args.dataset == 'cc3m'):
            classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',  'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
            'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

           
            kg_dict = defaultdict()
            
            kg_dict["person"] = {"Facial features": ["Eye color", "Nose shape", "Mouth size"], "Body shape": ["Height", "Weight", "Body type (e.g., athletic, slim, curvy)"], "Hair style": ["Hair length", "Hair color", "Hair texture"], "Clothing": ["Fashion sense", "Color preferences", "Clothing fit"], "Gait (walking style)": ["Speed", "Stride length", "Foot placement"]}
            kg_dict["bicycle"] = {"Two wheels": ["Round tires", "Spoked design", "Air or solid"], "Frame": ["Metal structure", "Geometric shape", "Tubing material"], "Pedals": ["Foot platforms", "Gear-driven mechanism", "Clip-in systems"], "Handlebars": ["Steering control", "Height adjustment", "Grip design"], "Seat": ["Padded surface", "Adjustable position", "Multiple styles"]}
            kg_dict["car"] = {"Four wheels": ["Two front wheels", "Two rear wheels", "Aligned in a square"], "Metallic body": ["Painted surface", "Various shapes", "Windows and windshield"], "Engine": ["Under the hood", "Exhaust system", "Cooling system"], "Headlights": ["Two on front", "Tail lights on rear", "Brake lights"], "Interior": ["Seats", "Steering wheel", "Dashboard"]}
            kg_dict["motorcycle"] = {"Two wheels": ["One front wheel", "One rear wheel", "Tyres"], "Engine": ["Exposed or covered", "Cylinders", "Exhaust system"], "Handlebars": ["One on each side", "Attached to the front wheel", "For steering"], "Seat": ["One or two", "For rider and passenger", "Attached to the frame"], "Silencer": ["At the end of the exhaust", "Muffles the sound", "Various shapes and designs"]}
            kg_dict["airplane"] = {"Wings": ["Long and narrow", "Fixed to the sides", "Curved on top"], "Tail": ["Vertical stabilizer", "Horizontal stabilizer", "Rudder"], "Propeller or jet engine": ["Spinning blades", "Attached to the front", "Provides thrust"], "Fuselage": ["Long and tube-like", "Windows for viewing", "Seats inside"], "Landing gear": ["Wheels for landing", "Retractable during flight", "Supports the aircraft"]}
            kg_dict["bus"] = {"Large size": ["Long and wide", "Tall with multiple floors", "Can carry many passengers"], "Multiple Windows": ["Large and rectangular", "Vertical arrangement", "Provide viewing"], "Doors": ["Multiple entry/exit points", "Along the side and front", "Sliding or swinging"], "Wheels": ["Four large tires", "Aligned in a square", "Support the weight"], "Roof": ["Flat or curved", "Often painted with a company or route logo", "Can be seen from a distance"]}
            kg_dict["train"] = {"Long and narrow shape": ["Can be many cars connected", "Passenger or freight cars", "Shape for efficient travel"], "Several Wheels": ["Many on each car", "Aligned in a row", "Metal or rubber tires"], "Engine": ["At the front or back", "Powers the train", "Can be diesel, electric, or steam"], "Caboose": ["At the end of the train", "For conductor or crew", "Often has a distinct look"], "Passenger cars": ["Multiple cars in a row", "Windows for viewing", "Seats facing forward or backward"]}
            kg_dict["truck"] = {"Large size": ["Long and wide", "Tall with a high ground clearance", "Strong and sturdy design"], "Cab": ["For the driver and passengers", "Usually with windows all around", "Attached to the truck bed"], "Truck bed": ["Open area for cargo", "Can be covered or uncovered", "Can be flat or with a lift gate"], "Wheels": ["Large and durable", "Aligned in a square or rectangle", "Often with off-road tires"], "Exhaust system": ["Comes from the engine", "Often with a distinct sound", "Can be single or dual pipes"]}
            kg_dict["boat"] = {"Streamlined hull": ["Curved design", "Smooth passage", "Reduced drag"], "Sleek sail": ["Wind capture", "Propulsion", "Angling adjustments"], "Sturdy oars": ["Rowing support", "Maneuverability", "Shore exploration"], "Navigable steering": ["Helm control", "Directional stability", "Safe passage"], "Seaworthy construction": ["Durable materials", "Watertight integrity", "Buoyancy maintenance"]}
            kg_dict["traffic light"] = {"Red, yellow, green lights": ["Color-coded signals", "Signal order", "Traffic regulation"], "Octagonal shape": ["Eight-sided form", "Recognizable design", "Unique geometry"], "Pedestrian signal": ["Crosswalk indicator", "Countdown timer", "WALK/DON'T WALK"], "Post-mounted installation": ["Roadside placement", "Vertical orientation", "Adjustable height"], "Lens cover": ["Protective shield", "Color refraction", "Focused illumination"]}
            kg_dict["fire hydrant"] = {"Red color": ["Distinctive hue", "High visibility", "Symbol of safety"], "Round shape": ["Sturdy construction", "Efficient flow", "Recognizable design"], "Valve control": ["Water release", "Pressure regulation", "Firefighting tool"], "Post installation": ["Ground stability", "Roadside placement", "Easy accessibility"], "Identification markings": ["Serial numbers", "Location indicators", "Maintenance records"]}
            kg_dict["stop sign"] = {"Red octagon": ["Eye-catching shape", "High contrast", "Universal symbol"], "Stop command": ["Bold lettering", "No ambiguity", "Traffic enforcement"], "Reflective coating": ["Enhanced visibility", "Night and day", "Inclement weather"], "Attached post": ["Ground mount", "Sturdy support", "Multiple heights"], "Roadside placement": ["Clear view", "Consistent spacing", "Strategic positioning"]}
            kg_dict["parking meter"] = {"Coin slot": ["Payment acceptance", "Coin variability", "Money collection"], "Digital display": ["Time indication", "Rate information", "Instruction prompts"], "Payment options": ["Coins only", "Card readers", "Mobile payment"], "Button controls": ["Operation initiation", "Duration selection", "Credit addition"], "LED lights": ["Active signals", "Enforcement indication", "User guidance"]}
            kg_dict["bench"] = {"Seating area": ["Resting spot", "Multiple slots", "People accommodation"], "Back support": ["Lumbar relief", "Comfort enhancement", "Design variation"], "Armrests": ["Upper body support", "Elbow rests", "Accessibility aid"], "Wooden construction": ["Durable material", "Natural appearance", "Weather-resistant"], "Metal frame": ["Sturdy structure", "Various finishes", "Long-lasting"]}
            kg_dict["bird"] = {"Flying creature": ["Winged locomotion", "Aerial agility", "Migration patterns"], "Feathered body": ["Plumage variety", "Insulation", "Camouflage"], "Beak": ["Wide variety of shapes", "Tool use", "Species identification"], "Taloned feet": ["Perching grip", "Ground traversal", "Predatory advantage"], "Reptilian eyes": ["Visual acuity", "Field of view", "Depth perception"]}
            kg_dict["cat"] = {"Feline body": ["Agile movement", "Sleek appearance", "Tail swishing"], "Long, thick whiskers": ["Touch sensitivity", "Facial expression", "Mood indication"], "Pointed ears": ["Alertness signal", "Hearing acuity", "Emotional response"], "Tail": ["Communication signal", "Balance aid", "Playful interaction"], "Retractable claws": ["Precise grip", "Self-defense", "Tree-climbing ability"]}
            kg_dict["dog"] = {"Bright eyes": ["Visual perception", "Pupil dilation", "Emotion conveyance"], "Fur coat": ["Insulation", "Breed variety", "Grooming needs"], "Four-legged stance": ["Sturdy posture", "Running prowess", "Adaptive locomotion"], "Wagging tail": ["Expression of joy", "Social interaction", "Communication tool"], "Black nose": ["Sensory power", "Breed distinction", "Exploration facilitator"]}
            kg_dict["horse"] = {"Equine build": ["Muscular physique", "Streamlined design", "Powerful locomotion"], "Flowing Mane and tail": ["Thick and luxurious", "Cascading movement", "Tail swishing"], "Powerful Legs": ["Strong hooves", "Sturdy bones", "Swift stride"], "Long neck": ["Display of strength", "Social signal", "Grazing posture"], "Expressive Eyes": ["Large and soft", "Intelligent gaze", "Sensitive expression"]}
            kg_dict["sheep"] = {"Woolly Coat": ["Soft fleece", "Dense coverage", "Variety of colors"], "Oval-shaped Body": ["Full figure", "Compact form", "Curved silhouette"], "Short Legs": ["Stout build", "Strong limbs", "Sturdy posture"], "Roman Nose": ["Curved profile", "Delicate features", "Inquisitive expression"], "Splayed Hooves": ["Pad-like structure", "Adaptive grip", "Mobility on uneven terrain"]}
            kg_dict["cow"] = {"Massive Frame": ["Bulky body", "Heavy muscles", "Broad back"], "Long Neck": ["Elegant curve", "Sweeping motion", "Grazing posture"], "Pendulous Udder": ["Milky secretion", "Soft texture", "Hanging shape"], "Deep Ribcage": ["Wide chest", "Strong heart", "Respiratory expansion"], "Horned Head": ["Curved horns", "Pointed tips", "Defensive weapon"]}
            kg_dict["elephant"] = {"Long trunk": ["Adaptive appendage", "Used for feeding and drinking", "Expression of emotions"], "Big ears": ["Heat regulation", "Hearing sensitivity", "Unique identification feature"], "Gray skin": ["Wrinkled appearance", "Camouflage in the wild", "Water-resistant"], "Tusks": ["Ivory teeth", "Used for defense and digging", "Indicate age and social status"], "Intelligent eyes": ["Expressive and emotive", "High level of cognitive abilities", "Empathy and understanding"]}
            kg_dict["bear"] = {"Massive Size": ["Imposing presence", "Bulky body", "Heavy build"], "Round Face": ["Soft features", "Small ears", "Wide snout"], "Padded Feet": ["Broad paws", "Curved claws", "Soft tread"], "Long Claws": ["Sharp retractable", "Curved extension", "Hooked tip"], "Shaggy Coat": ["Thick fur layer", "Winter insulation", "Mottled pattern"]}
            kg_dict["zebra"] = {"Striped Coat": ["Contrasting colors", "Vertical stripes", "Dazzling pattern"], "Erect Ears": ["Elongated shape", "Mobile movement", "Twitching response"], "Long Neck": ["Graceful curve", "Sweeping motion", "Reaching for food"], "Slender Legs": ["Swift stride", "Tapered limbs", "Hoofed feet"], "Tufted Tail": ["Flickering motion", "Bushy end", "Social signaling"]}
            kg_dict["giraffe"] = {"Long neck": ["Iconic feature", "Enables reaching leaves", "Social signaling"], "Tongue-like upper lip": ["Prehensile structure", "Used for grasping food", "Unique and expressive"], "Spotty pattern": ["Camouflage in the wild", "Individual variation", "Appeals to human aesthetics"], "Tall stature": ["Imposing presence", "Adaptation for browsing", "Easier to spot predators"], "Peculiar walk": ["Unique gait", "Legs on one side move together", "Fluid and graceful"]}
            kg_dict["backpack"] = {"Durable material": ["Sturdy and long-lasting", "Resistant to wear and tear", "Water-resistant or waterproof"], "Multiple pockets": ["Organized storage", "Easy access to essentials", "Different sizes for various items"], "Padded straps": ["Comfortable carry", "Reduces stress on shoulders", "Adjustable for custom fit"], "Zippered closure": ["Secure and easy access", "Protects contents from the elements", "Different types, such as waterproof or standard"], "Accessory loops": ["Bungee cord system", "Gear attachment points", "Key clip"]}
            kg_dict["umbrella"] = {"Waterproof material": ["Keeps users dry", "Resists moisture penetration", "Made of materials like nylon or polyester"], "Collapsible design": ["Easy to carry", "Compact for storage", "Convenient for travel or on-the-go use"], "Metal shaft": ["Strong and durable", "Provides structural support", "Can be used as a walking stick"], "Ribs and stretchers": ["Supports canopy structure", "Allows for ventilation and air circulation", "Protects against wind damage"], "Variety of sizes and shapes": ["Customized for different needs", "Includes options like compact, full-size, and golf umbrellas", "Features unique designs or patterns"]}
            kg_dict["handbag"] = {"Leather material": ["Soft texture", "Durable finish", "Smooth surface"], "Shoulder strap": ["Adjustable length", "Padded shoulder pad", "Crossbody or single strap"], "Zippered compartments": ["Secure closure", "Internal pockets", "Organizational features"], "Handles": ["Top handles", "Detachable straps", "Comfortable grip"], "Decorative details": ["Metal hardware", "Beaded embellishments", "Contrasting trims"]}
            kg_dict["tie"] = {"Neckwear Length": ["Standard size", "Long length", "Short style"], "Narrow Width": ["Slim design", "Wide stripes", "Thin lines"], "Knotted Design": ["Simple knot", "Four-in-hand knot", "Windsor knot"], "Textured Fabric": ["Silky material", "Woven pattern", "Smooth finish"], "Vibrant Colors": ["Bold hues", "Neutral tones", "Contrasting stripes"]}
            kg_dict["suitcase"] = {"Durable Construction": ["Solid frame", "Rigid shell", "Reinforced corners"], "Wheeled Mobility": ["Rolling bottom", "Pivoting wheels", "Retractable handle"], "Spacious Interior": ["Ample storage", "Compartmentalized space", "Zippered pockets"], "Exterior Pockets": ["Front organizers", "Side pouches", "Top compartments"], "Visible Identity": ["Colorful pattern", "Unique design", "Personalized tag"]}
            kg_dict["frisbee"] = {"Flat disc shape": ["Circular design", "Rim edge", "Concave center"], "Lightweight frame": ["Plastic composition", "Durable structure", "Easy throwing"], "Aerodynamic form": ["Streamlined profile", "Lift-generating curves", "Stable flight"], "Colorful design": ["Vibrant patterns", "Custom artwork", "Variety of styles"], "Throwing motion": ["Overhand release", "Underhand toss", "Forehand flick"]}
            kg_dict["skis"] = {"Long and narrow": ["Slender shape", "Tapered ends", "Curved sides"], "Metal edges": ["Sharp blades", "Sidecut definition", "Enhanced control"], "Adjustable bindings": ["Ankle support", "Heel lift", "Boot securement"], "Flexible construction": ["Soft core", "Bendable tails", "Responsive tips"], "Camber profile": ["Rising middle", "Contact points", "Pressure distribution"]}
            kg_dict["snowboard"] = {"Wide and flat": ["Extended surface", "Parabolic shape", "Stance platform"], "Riders bindings": ["Foot straps", "Ankle support", "Adjustable positioning"], "Flexible base": ["Soft material", "Torsional flex", "Edge-to-edge feel"], "Camber pattern": ["Rising middle", "Contact points", "Pressure distribution"], "Colorful design": ["Vibrant graphics", "Custom artwork", "Unique styles"]}
            kg_dict["sports ball"] = {"Round shape": ["Spherical geometry", "Uniform dimensions", "Symmetrical design"], "Colorful appearance": ["Vibrant hues", "Contrasting patterns", "Team-specific colors"], "Textured surface": ["Patterned grip", "Raised panels", "Smooth patches"], "Aerodynamic features": ["Streamlined shape", "Curved seams", "Lift-reducing design"], "Durable build": ["Synthetic cover", "Rubber bladder", "Reinforced stitching"]}
            kg_dict["kite"] = {"Soaring shape": ["Wing-like structure", "Tail assembly", "Aerodynamic curves"], "Colorful design": ["Vivid patterns", "Eye-catching hues", "Bright accents"], "Lightweight build": ["Rigid frame", "Flexible sail", "High-quality string"], "Tethered flight": ["Adjustable line", "Handheld control", "Wind-resistant anchors"], "Decorative details": ["Intricate embellishments", "Swirling tails", "Whimsical figures"]}
            kg_dict["baseball bat"] = {"Wooden construction": ["Solid grain", "Natural finish", "Rounded handle"], "Balanced weight": ["Evenly distributed", "Ideal swing momentum", "Well-proportioned"], "Curved profile": ["Tapered design", "Barrel shape", "Thin handle"], "Grip-enhancing details": ["Rubberized texture", "Padded contact points", "Raised lines"], "Branding and design": ["Logo placement", "Striped graphics", "Customized colors"]}
            kg_dict["baseball glove"] = {"Leather construction": ["Premium materials", "Durable finish", "Soft feel"], "Webbing and padding": ["Adjustable strap", "Protective cushioning", "Breathable mesh"], "Curved shape": ["Form-fitting design", "Segmented fingers", "Deep pocket"], "Velcro and straps": ["Secure fasteners", "Easy adjustment", "Custom fit"], "Colorful accents": ["Team-specific hues", "Stitched patterns", "Contrasting trim"]}
            kg_dict["skateboard"] = {"Wooden deck": ["Multi-ply construction", "Durable finish", "Grip tape application"], "Four wheels": ["PU cushioning", "Abec bearings", "Rubberized tread"], "T-shape design": ["Curved tail", "Pointed nose", "Concave surface"], "Artistic graphics": ["Customized artwork", "Vibrant colors", "Complex designs"], "Mobile and flexible": ["Lightweight build", "Easy maneuverability", "Adaptable riding styles"]}
            kg_dict["surfboard"] = {"Foam core": ["EPS or PU materials", "Strong and buoyant", "Lightweight support"], "Fiberglass coat": ["Durable and strong", "Smooth finish", "Resilient material"], "Colorful design": ["Vibrant patterns", "Eye-catching hues", "Custom graphics"], "Tail and fin setup": ["Different shapes and sizes", "Rear stabilizer", "Enhanced control"], "Leash attachment": ["Secure fastening", "Ankle or wrist strap", "Reliable connection"]}
            kg_dict["tennis racket"] = {"Graphite frame": ["Lightweight strength", "Durable material", "Responsive feel"], "Synthetic grip": ["Improved control", "Non-slip surface", "Cushioned texture"], "String pattern": ["Open or closed design", "String tension", "Power or control balance"], "Racket head": ["Oversized or standard", "Weight distribution", "Swing speed"], "Brand logo placement": ["On string pattern", "Striped graphics", "Custom color schemes"]}
            kg_dict["bottle"] = {"Glass construction": ["Transparent material", "Fragile nature", "Recyclable"], "Slim shape": ["Elegant design", "Compact form", "Easy grip"], "Opening and closure": ["Screw-top lid", "Cork stopper", "Twist-off cap"], "Label and branding": ["Logo placement", "Colorful design", "Informative text"], "Cork stopper": ["Wooden plug", "Air tight seal", "Moisture control"]}
            kg_dict["wine glass"] = {"Crystal material": ["Lead-free clarity", "Elegant appearance", "Durable construction"], "Stem and base": ["Balanced design", "Secure grip", "Stable footing"], "Bowl shape": ["Wide mouth", "Tapered sides", "U-shaped rim"], "Slightly curved rim": ["Delicate edge", "Gentle slope", "Comfortable lips"], "Glass container": ["Transparent material", "Fragile feel", "Thin appearance"]}
            kg_dict["cup"] = {"Ceramic construction": ["Durable and break-resistant", "Heat-retaining properties", "Variety of shapes and sizes"], "Handle and grip": ["Comfortable hold", "Secure attachment", "Ergonomic design"], "Sturdy base": ["Wide support", "Secure footing", "Even balance"], "Printed design": ["Patterned or solid colors", "Custom graphics", "Brand or logo placement"], "Lid and drinking opening": ["Snug-fitting cover", "Spill-resistant design", "Convenient sipping size"]}
            kg_dict["fork"] = {"Multiple tines": ["Evenly spaced", "Curved prongs", "Uniform length"], "Thin, flat shape": ["Delicate design", "Food-safe material", "Easy piercing"], "Sturdy construction": ["Durable metal", "Comfortable grip", "Rust-resistant finish"], "Curved handle": ["Ergonomic fit", "Smooth contours", "Balanced weight"], "Gripped by hands": ["Food manipulation", "Efficient eating", "Versatile use"]}
            kg_dict["knife"] = {"Sharp blade": ["Razor-like edge", "Cutting ability", "Fine point"], "Metal handle": ["Solid grip", "Weighted balance", "Comfortable use"], "Straight edge": ["Uninterrupted line", "Precise cuts", "Symmetrical design"], "Pointed tip": ["Penetrating strength", "Fine accuracy", "Easy insertion"], "Balanced and well-made": ["Even weight distribution", "High-quality craftsmanship", "Durable and long-lasting"]}
            kg_dict["spoon"] = {"Long handle": ["Easy grip", "Curved shape", "Comfortable hold"], "Shallow bowl": ["Narrow depth", "Wide mouth", "Flat bottom"], "Metal material": ["Shiny surface", "Durable construction", "Easy cleaning"], "Curved edge": ["Gentle slope", "Smooth transition", "Precise shape"], "Efficient scooping": ["Effective food transfer", "Smooth and effortless motion", "Used for eating"]}
            kg_dict["bowl"] = {"Curved shape": ["Smooth edges", "Rounded bottom", "Fluid lines"], "Hollow center": ["Open interior", "Empty space", "Nested design"], "Rimmed edge": ["Thin border", "Supportive lip", "Gentle slope"], "Sturdy material": ["Durable ceramic", "Heavy glass", "Lightweight plastic"], "Decorative pattern": ["Intricate design", "Vibrant colors", "Cultural influence"]}
            kg_dict["banana"] = {"Yellow and curved": ["Bright and cheerful", "Slightly bent shape", "Thin and flexible peel"], "Fruit and edible": ["Botanical classification", "Nutritious and delicious", "Versatile in culinary uses"], "Tapered and pointy": ["Gradually narrowing", "Sharp and defined tip", "Contrasting texture"], "Peel and protective layer": ["Thin and easily removed", "Serves as packaging", "Unique texture and aroma"], "Cluster and growing pattern": ["Groups on a stem", "Develop in stages", "Varying maturity levels"]}
            kg_dict["apple"] = {"Red and round": ["Vibrant and bold color", "Perfectly spherical shape", "Smooth and shiny skin"], "Fruit and edible": ["Botanical classification", "Nutritious and delicious", "Versatile in culinary uses"], "Stem and base": ["Attaches to the tree", "Remains on the fruit", "Serves as an identifying feature"], "Waxy and protective skin": ["Thin and durable layer", "Repels water and germs", "Maintains fruit freshness"], "Lighter core": ["Juicy consistency", "Tart taste", "Sweet aroma"]}
            kg_dict["sandwich"] = {"Bread and filling": ["Two slices of bread", "Assorted fillings", "Balanced proportions"], "Layers and ingredients": ["Stacked components", "Various meats, cheeses, and vegetables", "Customizable to preferences"], "Structured and organized": ["Neat and even arrangement", "Orderly presentation", "Easily portable and shareable"], "Colorful and appetizing": ["Vibrant and fresh ingredients", "Visually appealing", "Reflects diverse flavors"], "Flat shape": ["Uniform thickness", "Compressed edges", "Rounded corners"]}
            kg_dict["orange"] = {"Round and juicy": ["Perfectly spherical shape", "Plump and full appearance", "Moist and tender flesh"], "Citrus and edible": ["Belongs to the citrus family", "Nutritious and refreshing", "Versatile in culinary uses"], "Peel and zest": ["Thin and easily removed", "Aromatic zest adds flavor", "Unique texture and aroma"], "Segments and pulp": ["Divided fruit sections", "Juicy and tender pulp", "Easy to eat and share"], "Bright orange color": ["Vibrant and attractive", "Sunny appearance", "Shiny skin"]}
            kg_dict["broccoli"] = {"Green color": ["Rich hue", "Vibrant tones", "Deep shades"], "Floret arrangement": ["Compact structure", "Orderly pattern", "Dense clusters"], "Stalk structure": ["Solid consistency", "Firm texture", "Smooth surface"], "Tree-like shape": ["Branching branches", "Leafy crowns", "Vertical growth"], "Fibrous texture": ["Crunchy consistency", "Rough surface", "Chewy material"]}
            kg_dict["carrot"] = {"Orange color": ["Rich hue", "Vibrant tones", "Deep shades"], "Tapered shape": ["Narrow tip", "Widening toward base", "Pointed tip"], "Smooth skin": ["Unblemished surface", "Glossy finish", "Easy-to-peel"], "Straight texture": ["Uniform thickness", "Firm consistency", "Long, cylindrical shape"], "Green top": ["Leafy cover", "Soft shoots", "Fresh appearance"]}
            kg_dict["hot dog"] = {"Meaty appearance": ["Sausage-like shape", "Coarse texture", "Uniform thickness"], "Hot dog Skin": ["Crisp surface", "Snappy feel", "Smooth finish"], "Bread casing": ["Fluffy bun", "Seeded top", "Soft interior"], "Condiments": ["Colorful toppings", "Balanced distribution", "Artistic design"], "Bun": ["Plain bun", "Sesame-seed bun", "Sweet bun"]}
            kg_dict["pizza"] = {"Round shape": ["Perfect circle", "Even dimensions", "Flat surface"], "Crust": ["Thin crust", "Crispy edges", "Chewy center"], "Toppings": ["Assorted ingredients", "Artistic design", "Balanced distribution"], "Sauce": ["Rich red color", "Smooth consistency", "Slightly glossy finish"], "Cheese": ["Melted mozzarella", "Gooey texture", "Even layer"]}
            kg_dict["donut"] = {"Round shape": ["Perfect circle", "Symmetrical design", "Even dimensions"], "Glazed surface": ["Glittering sheen", "Sweet coating", "Smooth finish"], "Soft interior": ["Fluffy texture", "Lightly sweetened", "Moist consistency"], "Hole in center": ["Donut shape", "Airy feel", "Evenly spaced"], "Sprinkle decoration": ["Colorful specks", "Fun design", "Even distribution"]}
            kg_dict["cake"] = {"Layered structure": ["Stacked tiers", "Evenly divided", "Smooth transitions"], "Frosting appearance": ["Decorative swirls", "Silky texture", "Glossy finish"], "Fruit toppings": ["Colorful garnish", "Fresh presentation", "Balanced arrangement"], "Base board": ["Serving plate", "Cake stand", "Presentation tray"], "Crumbs": ["Moist texture", "Dense consistency", "Soft bite"]}
            kg_dict["chair"] = {"Seat and backrest": ["Cushioned surface", "Curved support", "Upholstered design"], "Four legs": ["Sturdy support", "Angular structure", "Grounded stability"], "Armrests": ["Additional support", "Padded comfort", "Decorative detail"], "Swivel mechanism": ["No swivel", "Pivoting base", "Enhanced mobility"], "Wheels": ["Glides", "Locking mechanism", "No wheels"]}
            kg_dict["couch"] = {"Long and wide shape": ["Extended seating", "Ample space", "Comfortable fit"], "Multiple cushions": ["Pillow-like support", "Soft upholstery", "Custom arrangements"], "Armrests": ["Padded edges", "Resting surfaces", "Decorative accents"], "Pillow back": ["Cushioned headrest", "Lumbar support", "Comfort enhancer"], "Fabric upholstery": ["Patterned textiles", "Solid colors", "Natural fibers"]}
            kg_dict["potted plant"] = {"Container": ["Pottery vessel", "Terracotta pot", "Plastic planter"], "Green foliage": ["Leafy canopy", "Vibrant colors", "Varied textures"], "Stem and branches": ["Upright growth", "Branching structure", "Twisting form"], "Roots (hidden)": ["Underground system", "Nutrient absorption", "Anchoring support"], "Soil and medium": ["Earthy foundation", "Nutrient-rich mix", "Drainage properties"]}
            kg_dict["bed"] = {"Flat surface": ["Comfortable lying", "Even support", "Spacious design"], "Soft mattress": ["Plush cushioning", "Pressure relief", "Temperature regulation"], "Cozy blankets": ["Warm coverage", "Variety of fabrics", "Stylish patterns"], "Pillowy pillows": ["Supportive cushioning", "Adjustable positioning", "Hypoallergenic materials"], "Sturdy frame": ["Durable construction", "Easy assembly", "Versatile design"]}
            kg_dict["dining table"] = {"Solid base": ["Sturdy legs", "Balanced structure", "Weighty design"], "Smooth surface": ["Easy cleaning", "Comfortable gliding", "Wood or glass finish"], "Expandable size": ["Adjustable length", "Additional leaves", "Compact storage"], "Matching chairs": ["Complementary design", "Comfortable seating", "Coordinated upholstery"], "Ample space": ["Seating capacity", "Storage options", "Versatile functionality"]}
            kg_dict["toilet"] = {"Porcelain bowl": ["Ceramic construction", "Smooth surface", "Easy cleaning"], "Chrome plated handle": ["Shiny finish", "Durable coating", "Easy grip"], "Water-saving flush": ["Efficient design", "Low-flow mechanism", "Environmentally friendly"], "Soft-close seat": ["Quiet operation", "Comfortable use", "Hygienic design"], "Washlet attachment": ["Advanced features", "Heated seat", "Bidet function"]}
            kg_dict["tv"] = {"Flat screen": ["Thin design", "Lightweight", "Modern look"], "LED display": ["Crisp resolution", "High contrast", "Energy efficient"], "Curved shape": ["Immersive viewing", "Design aesthetic", "Comfortable focus"], "Smart functionality": ["Internet connectivity", "Streaming services", "Voice control"], "Wall-mounted setup": ["Space-saving placement", "Secure installation", "Minimalist appeal"]}
            kg_dict["laptop"] = {"Sleek style": ["Thin profile", "Lightweight build", "Stylish appearance"], "Touchscreen display": ["Clear resolution", "Multi-touch gestures", "Integrated camera"], "Backlit keyboard": ["Easy typing", "Adjustable brightness", "Convenient use"], "Multiple ports": ["USB connectivity", "HDMI output", "Card reader"], "Ergonomic design": ["Curved key layout", "Wrist support", "Comfortable use"]}
            kg_dict["mouse"] = {"Ergonomic shape": ["Comfortable grip", "Natural hand position", "Reduced strain"], "Smooth scroll wheel": ["Fluid motion", "Precise tracking", "Multiple scrolling options"], "Tailored buttons": ["Customizable layout", "Easy-to-press buttons", "Programmable functions"], "Optical sensor": ["High accuracy", "Smooth movement", "Works on various surfaces"], "Wireless connectivity": ["Cordless operation", "Freedom of movement", "Easy pairing"]}
            kg_dict["remote"] = {"Compact size": ["Easy grip", "Portable design", "Convenient carrying"], "User-friendly buttons": ["Clear labeling", "Raised icons", "Well-spaced layout"], "Remote control range": ["Line of sight use", "Obstacle interference", "Optimal operating distance"], "Infrared technology": ["Line-of-sight control", "Fast response", "Universal compatibility"], "Remote finder function": ["Locator feature", "Audible signal", "Visual indicator"]}
            kg_dict["keyboard"] = {"QWERTY layout": ["Alphabetical order", "Specific key arrangement", "Typing efficiency"], "Numeric keypad": ["Number buttons", "Calculator functionality", "Data entry ease"], "Key markers": ["Raised bumps", "Tactile feedback", "Visually impaired assistance"], "LED indicators": ["Backlighting", "Visual feedback", "Low-light situations"], "Ergonomic design": ["Curved key layout", "Wrist support", "Comfortable use"]}
            kg_dict["cell phone"] = {"Sleek design": ["Thin profile", "Modern aesthetic", "Comfortable hold"], "Touchscreen interface": ["Responsive display", "Gesture controls", "Vivid graphics"], "High-resolution camera": ["Clear images", "Video recording", "Wide angle"], "Personalized design": ["Variety of colors", "Unique styles", "Matching accessories"], "Long battery life": ["All-day use", "Energy-saving mode", "Quick charging"]}
            kg_dict["microwave"] = {"Electronic appliance": ["Plastic casing", "Digital display", "Various buttons"], "Cooking device": ["Microwave radiation", "Heating food", "Defrosting function"], "Door with handle": ["Transparent window", "Interior light", "Easy access"], "Rotating turntable": ["Smooth surface", "Even heating", "t"], "Power cord and outlet": ["Plug-in design", "Wall-mounted socket", "Grounded for safety"]}
            kg_dict["oven"] = {"Cooking appliance": ["Metal exterior", "Insulated interior", "Heating elements"], "Baking chamber": ["Oven racks", "Broiling element", "Adjustable temperature"], "Door with window": ["Glass panel", "Viewing food", "Lock mechanism"], "Dual function": ["Convection cooking", "Standard baking", "Energy efficiency"], "Control panel with options": ["Temperature settings", "Timer functions", "Light indicators"]}
            kg_dict["toaster"] = {"Square shape": ["Standard design", "Compact size", "Easy placement"], "Toaster slots": ["Evenly spaced", "Adjustable width", "Accommodate bread sizes"], "Browning control": ["Shade settings", "Temperature adjustment", "Customize toast color"], "Cool touch exterior": ["Safe operation", "Insulated body", "Reduced burn risk"], "Cord": ["Tidy appearance", "Wrap-around space", "Organized storage solution"]}
            kg_dict["sink"] =  {"Porcelain material": ["Ceramic composition", "Durable surface", "Easy cleaning"], "Stainless steel material": ["Rust-resistant", "Modern look", "Solid construction"], "Water source": ["Faucet handles", "Cold and hot water", "Mixing valve"], "Faucet": ["Handle configuration", "Spout style", "Water dispensing"], "Drain system": ["P-trap assembly", "Water disposal", "Catches debris"]}
            kg_dict["refrigerator"] = {"Large appliance": ["Bulky structure", "Standing unit", "Home necessity"], "Cooling system": ["Compressor function", "Cold air circulation", "Temperature control"], "Freezer compartment": ["Ice maker", "Frozen storage", "Separate drawer"], "Door shelves": ["Adjustable compartments", "Storage organization", "Fresh food storage"], "Intricate design elements": ["Color variations", "Stainless steel finish", "Door handle design"]}
            kg_dict["book"] = {"Hard cover": ["Protective shell", "Durable binding", "Decorative jacket"], "Soft cover": ["Flexible pages", "Paperback binding", "Lightweight"], "Paper pages": ["Printed text", "Smooth surface", "Various sizes"], "Binding": ["Spine adhesion", "Edge stitching", "Page attachment"], "Cover design": ["Illustrations", "Typography", "Color variations"]}
            kg_dict["clock"] = {"Hands or digits": ["Minute and hour hands", "Numerical display", "Analog or digital"], "Circular face": ["Round dial", "Decorative elements", "Various sizes"], "Moving parts": ["Gears and springs", "Mechanical operation", "Quartz or manual"], "Dial markers": ["Minute and hour indicators", "Numerical or Roman numerals", "Subdials or complications"], "Base or case": ["Framework support", "Material composition", "Design accents"]}
            kg_dict["vase"] = {"Glass material": ["Transparent", "Fragile", "Shiny surface"], "Ceramic material": ["Earthenware", "Glazed finish", "Durable"], "Vase shape": ["Curved silhouette", "Tapering form", "Various sizes"], "Floral arrangement": ["Stem holder", "Flower display", "Vase purpose"], "Decorative pattern": ["Hand-painted", "Cultural motifs", "Colorful designs"]}
            kg_dict["scissors"] = {"Sharp blades": ["Cutting edges", "Pointed tips", "Metallic material"], "Pivot point": ["Middle joint", "Swivel mechanism", "Allows motion"], "Handles": ["Grip surfaces", "Finger loops", "Plastic or rubber material"], "Spring action": ["Tension mechanism", "Opening and closing", "Assists cutting"], "Lock mechanism": ["Safety feature", "Prevents accidents", "Secures blades"]}
            kg_dict["teddy bear"] = {"Plush fur": ["Soft and cuddly", "Various textures", "Composed of fabric"], "Round body": ["Stuffed with filler", "Huggable contours", "Defined by seams"], "Button eyes": ["Embroidered detail", "Limited movement", "Attached patches"], "Smiling face": ["Stitched features", "Expressive mouth", "Warming presence"], "Floppy limbs": ["Dangling arms", "Bendable joints", "Comforting weight"]}
            kg_dict["hair drier"] = {"Long nozzle": ["Flexible tubing", "Narrow opening", "Adjustable angle"], "Heat settings": ["Temperature control", "High and low options", "Variable intensity"], "Cold shot button": ["Instant cooling", "Refreshing finish", "Temporary switch"], "Handheld design": ["Easy grip handle", "Lightweight build", "Corded or cordless"], "Dual voltage": ["Adaptable power", "Suitable for travel", "Wide compatibility"]}
            kg_dict["toothbrush"] = {"Small head": ["Compact bristles", "Narrow neck", "Tapered end"], "Bristle arrangement": ["Cross-hatch pattern", "Densely packed", "Various lengths"], "Angled handle": ["Ergonomic grip", "Comfortable hold", "Balanced weight"], "Soft bristles": ["Gentle on gums", "Smooth texture", "Less abrasive"], "Replaceable head": ["Easy attachment", "Regular maintenance", "Hygienic upgrade"]}
                    
        return classes, kg_dict
    
    def get_kg_emb (self, kg_dict, row=10, col=3):
        
        ## subKE
        lst_tokens = []
        for kg in kg_dict.values():
            lst_subtokens = []
            for subke in kg.values():
                subkg_tokens = clip.tokenize(subke)
                lst_subtokens.append(subkg_tokens)
            sub_stacked_tensors = torch.stack(lst_subtokens)
            lst_tokens.append(sub_stacked_tensors)
        
        stacked_tensors = torch.stack(lst_tokens)
        
        ## KE
        lst_tensors = [] 
        for kg in kg_dict.values():
            lst_subke_keys = []
            for subke_key in kg.keys():
                lst_subke_keys.append(clip.tokenize(subke_key))
            subke_keys_tensor = torch.stack(lst_subke_keys)
            lst_tensors.append(subke_keys_tensor.squeeze(1))

        keys_stacked_tensors = torch.stack(lst_tensors)

        print('shapes: ', stacked_tensors.shape, keys_stacked_tensors.shape)
        return stacked_tensors, keys_stacked_tensors 
    
    def kgemb_bert (self, kg_dict, tokenizer, row = 10, col = 3):

       
        # lst_tokens = {}
        # lst_input_ids = []
        # lst_attention_masks = []
        # for kg in kg_dict.values():
        #     kg_tokens = tokenizer.batch_encode_plus(kg, max_length=100, padding='max_length', truncation=True, return_tensors='pt')
        #     lst_input_ids.append(kg_tokens['input_ids'])
        #     lst_attention_masks.append(kg_tokens['attention_mask'])

        
        # stacked_tensors_ids = torch.stack(lst_input_ids)
        # stacked_tensors_masks = torch.stack(lst_attention_masks)
        
        # lst_tokens['input_ids'] = stacked_tensors_ids
        # lst_tokens['attention_mask'] = stacked_tensors_masks
        # print('shape: ', stacked_tensors_ids.shape, stacked_tensors_masks.shape)

        ## KE
        lst_ke_tokens = {}
        lst_subke_id_tokens = []
        lst_subke_mask_tokens = []
        for kg in kg_dict.values():
            lst_subke_input_ids = []
            lst_subke_attention_masks = []
            lst_subke = []
            for subke_key in kg.keys():
                lst_subke.append(subke_key)

            subke_tokens = tokenizer.batch_encode_plus(lst_subke, max_length=100, padding='max_length', truncation=True, return_tensors='pt')
            lst_subke_input_ids.append(subke_tokens['input_ids'])
            lst_subke_attention_masks.append(subke_tokens['attention_mask'])
           
            lst_subke_id_tokens.append(torch.stack(lst_subke_input_ids))
            lst_subke_mask_tokens.append(torch.stack(lst_subke_attention_masks))
        
        lst_ke_tokens['input_ids'] = torch.stack(lst_subke_id_tokens).squeeze(1)
        lst_ke_tokens['attention_mask'] = torch.stack(lst_subke_mask_tokens).squeeze(1) 
        
        
        ##subKE
        lst_subke_tokens = {}
        lst_subke_id_tokens = []
        lst_subke_mask_tokens = []
        for kg in kg_dict.values():
            lst_subke_input_ids = []
            lst_subke_attention_masks = []
            for subke_key in kg.values():
                subke_tokens = tokenizer.batch_encode_plus(subke_key, max_length=100, padding='max_length', truncation=True, return_tensors='pt')
                lst_subke_input_ids.append(subke_tokens['input_ids'])
                lst_subke_attention_masks.append(subke_tokens['attention_mask'])
            
            lst_subke_id_tokens.append(torch.stack(lst_subke_input_ids))
            lst_subke_mask_tokens.append(torch.stack(lst_subke_attention_masks))
        
        lst_subke_tokens['input_ids'] = torch.stack(lst_subke_id_tokens)
        lst_subke_tokens['attention_mask'] = torch.stack(lst_subke_mask_tokens)
        
        print('shape subke and ke: ',   lst_subke_tokens['input_ids'].shape, lst_ke_tokens['input_ids'].shape)

        return lst_ke_tokens, lst_subke_tokens
    

    def kgemb_bert_flickr (self, kg_list, tokenizer):

        lst_tokens = {}
        lst_input_ids = []
        lst_attention_masks = []
        for kg in kg_list:
            kg_tokens = tokenizer.batch_encode_plus(kg, max_length=100, padding='max_length', truncation=True, return_tensors='pt')
            lst_input_ids.append(kg_tokens['input_ids'])
            lst_attention_masks.append(kg_tokens['attention_mask'])

        
        stacked_tensors_ids = torch.stack(lst_input_ids)
        stacked_tensors_masks = torch.stack(lst_attention_masks)
        
        lst_tokens['input_ids'] = stacked_tensors_ids
        lst_tokens['attention_mask'] = stacked_tensors_masks
        print('shape: ',  lst_tokens['input_ids'].shape,  lst_tokens['attention_mask'].shape)

        return lst_tokens, None
    


    