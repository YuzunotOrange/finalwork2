// Add some retro flair with a scrolling marquee
const marquee = document.createElement("marquee");
marquee.innerHTML = "Choose your interesting things!!";
marquee.direction = "left";
marquee.behavior = "alternate";
document.body.appendChild(marquee);