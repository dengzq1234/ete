import { view } from "./gui.js";
import { cartesian_shifted } from "./draw.js";

export { draw_pixi, clear_pixi, apps };


// Globals
const apps = {};  // pixi "applications" per container
let app;
let textures;
let textures_loaded = false;

// Codes for amino acids, nucleotides (DNA and RNA), and gaps.
// See for example https://en.wikipedia.org/wiki/FASTA_format#Sequence_representation
const aa = [
    'A', 'R', 'N',
    'D', 'C', 'Q',
    'E', 'G', 'H',
    'I', 'L', 'K',
    'M', 'F', 'P',
    'S', 'T', 'W',
    'Y', 'V', 'B',
    'Z', 'X', '.',
    '-'
];

const gradients = [
    'a', 'b', 'c',
    'd', 'e', 'f',
    'g', 'h', 'i',
    'j', 'k', 'l',
    'm', 'n', 'o',
    'p', 'q', 'r', 
    's', 't', '-', 
    'x', 'z'
];

// Load texture atlas
PIXI.Loader.shared
    .add(aa.map(a => ({name: `aa_notext_${a}`, url: `images/aa_notext/${a}.png`})))
    .add(aa.map(a => ({name: `aa_text_${a}`,   url: `images/aa_text/${a}.png`})))
    .add(gradients.map(gradient => ({name: `gradients_${gradient}`, url: `images/gradients/${gradient}.png`})))
    .add("block", "images/block.png")
    .load(() => {
        const resources = PIXI.Loader.shared.resources;  // shortcut

        const textures_notext = {};
        const textures_text = {};
        const textures_gradients = {};
        
        for (const a of aa) {
            textures_notext[a] = resources[`aa_notext_${a}`].texture;
            textures_text[a] = resources[`aa_text_${a}`].texture;
        }
        for (const gradient of gradients) {
            textures_gradients[gradient] = resources[`gradients_${gradient}`].texture;
        }

        textures = {
            aa_notext: textures_notext,
            aa_text: textures_text,
            gradients:textures_gradients,
            shapes: {
                block: resources.block.texture,
            }
        }

        textures_loaded = true;
    });

// Remove all items from app.stage.
function clear_pixi(container) {
    app = apps[container.id];

    if (app)
        app.stage.children = [];
}


// Create PIXI.Application object for the given container if it
// doesn't exist and keep it in apps, draw the given items (adding
// them to app.stage) and return the app.view.
function draw_pixi(container, items, tl, zoom) {
    app = apps[container.id] = apps[container.id] || new PIXI.Application({
        transparent: true,
        resolution: 1,
    });

    // Resize canvas based on container
    app.renderer.resize(container.clientWidth, container.clientHeight);

    // Remove all items from stage
    app.stage.children = [];

    if (textures_loaded && items.length > 0)
        draw(items, tl, zoom);

    return app.view;
}


// Add items to app.stage.
function draw(items, tl, zoom) {
    items.forEach(seq => {
        const [ el, box ] = [ seq[0], seq[1] ];
        const type = el.split("-")[1]
        const [ zx, zy ] = [ zoom.x, zoom.y ];
        if (["aa_notext", "aa_text", "nt_notext", "nt_text", 'gradients'].includes(type))
            draw_msa(seq[2], type, box, tl, zx, zy);
        else
            draw_shape(type, box, tl, zx, zy)
    })
}


// Add the given sprite to app.stage, placing it according to the
// requested box, top-left corner, and taking into account the zoom.
function addSprite(sprite, box, tl, zx, zy, tooltip) {
    const [x, y, dx, dy] = box;

    if (view.drawer.type === "rect") {
        sprite.x = zx * (x - tl.x);
        sprite.y = zy * (y - tl.y);
    }
    else {  // "circ"
        const p = cartesian_shifted(sx + dx/2, sy + dy/2, tl, zx);
        sprite.x = p.x;
        sprite.y = p.y;

        sprite.anchor.set(0.5, 0.5);
        sprite.rotation = y + dy/2;
    }

    sprite.width = zx * dx;
    sprite.height = zy * dy;

    if (tooltip)
        sprite.accessibleTitle = tooltip;

    app.stage.addChild(sprite);
}


// Add a lot of sprites to app.stage, one per letter in the given sequence.
function draw_msa(sequence, type, box, tl, zx, zy) {
    const MIN_MSA_POSWIDTH = 12;

    const [x, y, dx, dy] = box;

    const sdx = dx / sequence.length;  // single sprite dx

    if (view.aligned.adjust_zoom)
        view.aligned.max_zoom = MIN_MSA_POSWIDTH / sdx;
    else
        view.aligned.max_zoom = undefined;

    sequence.split("").forEach((s, i) => {
        if (s != "-") {
            const sprite = new PIXI.Sprite(textures[type][s]);
            const tooltip = `Residue: ${s}\nPosition: ${i}`;
            addSprite(sprite, [x + i * sdx, y, sdx, dy], tl, zx, zy, tooltip);
        }
    });
}


// Add a shape sprite to app.stage.
function draw_shape(shape, box, tl, zx, zy) {
    const sprite = new PIXI.Sprite(textures.shapes[shape])
    addSprite(sprite, box, tl, zx, zy);
}
