"""
Zelflerende Flappy Bird met neat-python en pygame.

Gebaseerd op de tutorial "AI plays Flappy Bird" van Tech With Tim.
Aangepast voor de opdracht: pipe gap 160, base- en pipe-snelheid 5, 100 fps,
doel: een vogel die een score van 1000+ haalt in zo min mogelijk generaties
met een zo klein mogelijke populatie.

Gebruik:
    python flappy_neat.py           -> eerst trainen, daarna eindspel met het beste vogeltje
    python flappy_neat.py speel     -> alleen eindspel met het opgeslagen beste vogeltje (winner.pkl)

Bestanden die nodig zijn in dezelfde map:
    config-feedforward.txt   (NEAT-instellingen, o.a. populatiegrootte)
    imgs/                    (pipe.png, bg.png, bird1-3.png, base.png)

OVERZICHT VAN DE EIGEN WIJZIGINGEN (t.o.v. de tutorialcode)
-----------------------------------------------------------
1. Speelveld: PIPE_GAP = 160, SNELHEID = 5 (pipes en base), FPS = 100.
2. Bugfix: in de tutorialcode werden vogels uit de lijsten verwijderd terwijl er
   over die lijsten werd gelopen. Daardoor werd er soms een vogel overgeslagen.
   Nu verzamelen we per frame eerst de dode vogels en verwijderen ze pas daarna.
3. Bugfix: de test of een pipe gepasseerd is gebruikte de variabele 'bird' die
   toevallig nog over was uit een eerdere lus. Dat gaf een fout als alle vogels
   dood waren. Nu gebruiken we de vaste constante BIRD_X.
4. Snelheid: de maskers van de pipes en vogels worden maar een keer aangemaakt
   (niet elke frame opnieuw) en er is een snelle rechthoek-voorcontrole voordat de
   precieze botsingstest met maskers wordt gedaan.
5. Trainen zonder scherm: met TOON_SCHERM = False tekent het spel niets en wacht
   het niet op de klok. Training gaat dan vele malen sneller.
6. Stoppen bij succes: bereikt een vogel SCORE_DOEL pipes, dan stopt de training
   direct en wordt die vogel als winnaar gebruikt.
7. Pickle: de winnaar wordt opgeslagen in winner.pkl en kan later weer geladen
   worden voor het eindspel (speel_met_beste_vogel).
8. Fitness function: zie de uitleg bij eval_genomes.
"""
import os
import sys
import pickle
import random

import pygame
import neat

pygame.font.init()

# ---------------------------------------------------------------------------
# INSTELLINGEN
# ---------------------------------------------------------------------------
WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730            # y-positie van de grond (bovenkant van de base)
FPS = 100              # frames per seconde (opdracht)
SNELHEID = 5           # snelheid van pipes en base in pixels per frame (opdracht)
PIPE_GAP = 160         # ruimte tussen boven- en onderpijp (opdracht)
BIRD_X = 230           # vogel beweegt alleen verticaal, x blijft vast
BIRD_START_Y = 350

# Aantal gepasseerde pipes waarbij de training stopt. De opdracht vraagt om
# 1000+, maar dat duurt lang. Zet dit tijdens het afstellen eventueel op 100
# en zet het voor de uiteindelijke test op 1000.
SCORE_DOEL = 1000

MAX_GENERATIES = 50    # maximum aantal generaties voordat het trainen stopt
TOON_SCHERM = True     # False = snel trainen zonder tekenen en zonder klok
RANDOM_SEED = None     # bijv. 1 zetten om runs eerlijk met elkaar te vergelijken
WINNER_BESTAND = "winner.pkl"

STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False     # rode lijnen van vogel naar de pipe tekenen (alleen handig voor uitleg)

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs", "bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird" + str(x) + ".png"))) for x in range(1, 4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")).convert_alpha())

# Maskers eenmalig aanmaken (scheelt veel rekentijd tijdens het trainen).
PIPE_TOP_IMG = pygame.transform.flip(pipe_img, False, True)
PIPE_BOTTOM_IMG = pipe_img
PIPE_TOP_MASK = pygame.mask.from_surface(PIPE_TOP_IMG)
PIPE_BOTTOM_MASK = pygame.mask.from_surface(PIPE_BOTTOM_IMG)
BIRD_MASKS = [pygame.mask.from_surface(img) for img in bird_images]

gen = 0  # teller voor het aantal generaties (alleen voor weergave)


class DoelBereikt(Exception):
    """
    Wordt opgegooid zodra een vogel SCORE_DOEL pipes heeft gehaald. Zo stoppen we
    de NEAT-training meteen en kunnen we het genome van de winnaar meegeven.
    """
    def __init__(self, genome):
        super().__init__("Doelscore bereikt")
        self.genome = genome


class Bird:
    """
    De vogel. Beweegt alleen omhoog en omlaag; de pipes komen naar hem toe.
    """
    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0  # kanteling in graden
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        """Laat de vogel springen (negatieve snelheid = omhoog)."""
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        """Beweeg de vogel een frame, met zwaartekracht en maximale valsnelheid."""
        self.tick_count += 1

        displacement = self.vel * self.tick_count + 0.5 * 3 * self.tick_count ** 2

        # maximale valsnelheid
        if displacement >= 16:
            displacement = 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:  # kantel omhoog
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:  # kantel omlaag
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        """Teken de vogel met vleugelanimatie en kanteling."""
        self.img_count += 1

        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        # bij een duikvlucht flapperen we niet
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self):
        """Masker van het huidige vogelplaatje (uit de cache, niet opnieuw berekend)."""
        return BIRD_MASKS[self.IMGS.index(self.img)]


class Pipe:
    """
    Een paar pipes (boven en onder) met een opening van PIPE_GAP pixels.
    """
    GAP = PIPE_GAP
    VEL = SNELHEID

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0

        self.PIPE_TOP = PIPE_TOP_IMG
        self.PIPE_BOTTOM = PIPE_BOTTOM_IMG
        self.breedte = PIPE_TOP_IMG.get_width()

        self.passed = False
        self.set_height()

    def set_height(self):
        """
        Kies willekeurig waar de opening begint (self.height = onderkant van de
        bovenpijp). self.bottom is de bovenkant van de onderpijp.
        """
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        """
        True als de vogel de pipe raakt. Eerst een goedkope controle: als de
        vogel horizontaal niet in de buurt van de pipe is, kan er geen botsing
        zijn. Pas daarna de precieze test met maskers.
        """
        vogel_breedte = bird.img.get_width()
        if bird.x + vogel_breedte < self.x or bird.x > self.x + self.breedte:
            return False

        bird_mask = bird.get_mask()
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(PIPE_BOTTOM_MASK, bottom_offset)
        t_point = bird_mask.overlap(PIPE_TOP_MASK, top_offset)

        return bool(b_point or t_point)


class Base:
    """De bewegende grond. Twee plaatjes achter elkaar die samen opschuiven."""
    VEL = SNELHEID
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(surf, image, topleft, angle):
    """Draai een plaatje om zijn middelpunt en teken het op het scherm."""
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)
    surf.blit(rotated_image, new_rect.topleft)


def draw_window(win, birds, pipes, base, score, gen, pipe_ind, toon_generatie=True):
    """
    Teken het hele scherm.
    :param birds: lijst met vogels die nog leven
    :param pipe_ind: index van de dichtstbijzijnde pipe (voor de rode lijnen)
    :param toon_generatie: False in het eindspel (daar is geen generatieteller)
    """
    win.blit(bg_img, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255, 0, 0), (bird.x + bird.img.get_width() / 2, bird.y + bird.img.get_height() / 2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width() / 2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255, 0, 0), (bird.x + bird.img.get_width() / 2, bird.y + bird.img.get_height() / 2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width() / 2, pipes[pipe_ind].bottom), 5)
            except IndexError:
                pass
        bird.draw(win)

    score_label = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    if toon_generatie:
        gen_label = STAT_FONT.render("Gens: " + str(gen), 1, (255, 255, 255))
        win.blit(gen_label, (10, 10))

        alive_label = STAT_FONT.render("Alive: " + str(len(birds)), 1, (255, 255, 255))
        win.blit(alive_label, (10, 50))

    pygame.display.update()


def dichtstbijzijnde_pipe(pipes):
    """
    Geeft de index (0 of 1) van de pipe die de vogel nu moet nemen. Is de vogel al
    voorbij de eerste pipe, dan kijkt hij naar de tweede.
    """
    if len(pipes) > 1 and BIRD_X > pipes[0].x + pipes[0].breedte:
        return 1
    return 0


def netwerk_invoer(bird, pipe):
    """
    De drie invoerwaarden voor het neurale netwerk:
      1. hoogte van de vogel
      2. afstand van de vogel tot de onderkant van de bovenpijp
      3. afstand van de vogel tot de bovenkant van de onderpijp
    """
    return (bird.y, abs(bird.y - pipe.height), abs(bird.y - pipe.bottom))


def eval_genomes(genomes, config):
    """
    Speelt een hele generatie. Elke genome in de populatie bestuurt een eigen
    vogel; alle vogels spelen tegelijk in hetzelfde spel (zelfde pipes).
    Aan het eind van de functie moet elk genome een .fitness hebben; NEAT
    gebruikt die om te bepalen wie zich mag voortplanten.

    FITNESS FUNCTION (de beloning, hoe hoger hoe beter):
      * +0,1 per frame dat de vogel leeft. Dit beloont overleven en zorgt dat
        vogels die verder komen altijd hoger scoren dan vogels die meteen sterven.
      * +5 voor elke pipe die de vogel passeert (voor alle levende vogels). Dit is
        de hoofdbeloning: het echte doel van het spel is door de pipes komen.
      * -1 als de vogel een pipe raakt. Dit straft botsen tegen een pipe zwaarder
        dan sterven door de grond of het plafond, zodat vogels leren de opening
        te zoeken in plaats van tegen een pipe aan te vliegen.
    De vogel stuurt 1 uitvoerwaarde aan: is die groter dan 0,5 dan springt hij.

    Stopt de generatie zodra alle vogels dood zijn, of gooit DoelBereikt op
    als de score SCORE_DOEL bereikt.
    """
    global gen
    gen += 1

    nets = []
    birds = []
    ge = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(BIRD_X, BIRD_START_Y))
        ge.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()

    while len(birds) > 0:
        if TOON_SCHERM:
            clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pipe_ind = dichtstbijzijnde_pipe(pipes)

        # 1. Elke levende vogel beslist en beweegt
        for i, bird in enumerate(birds):
            ge[i].fitness += 0.1
            bird.move()
            output = nets[i].activate(netwerk_invoer(bird, pipes[pipe_ind]))
            if output[0] > 0.5:
                bird.jump()

        base.move()

        # 2. Pipes bewegen en botsingen controleren
        dood = set()  # indices van vogels die in dit frame doodgaan
        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()

            for i, bird in enumerate(birds):
                if i not in dood and pipe.collide(bird):
                    ge[i].fitness -= 1
                    dood.add(i)

            if pipe.x + pipe.breedte < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < BIRD_X:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            for i, genome in enumerate(ge):
                if i not in dood:
                    genome.fitness += 5
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        # 3. Vogels die de grond raken of boven het scherm uitvliegen
        for i, bird in enumerate(birds):
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                dood.add(i)

        # 4. Pas nu de dode vogels uit de lijsten halen (niet tijdens het lopen erover)
        if dood:
            nets = [n for i, n in enumerate(nets) if i not in dood]
            ge = [g for i, g in enumerate(ge) if i not in dood]
            birds = [b for i, b in enumerate(birds) if i not in dood]

        # 5. Doelscore bereikt? Dan is een levende vogel onze winnaar.
        if score >= SCORE_DOEL and len(ge) > 0:
            raise DoelBereikt(ge[0])

        if TOON_SCHERM:
            draw_window(WIN, birds, pipes, base, score, gen, pipe_ind)


def run(config_file):
    """
    Train de populatie met NEAT en bewaar de winnaar met pickle.
    :param config_file: pad naar config-feedforward.txt
    :return: het winnende genome
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    try:
        winner = p.run(eval_genomes, MAX_GENERATIES)
        print("\nMaximaal aantal generaties bereikt zonder doelscore; beste genome wordt gebruikt.")
    except DoelBereikt as klaar:
        winner = klaar.genome
        print("\nDoelscore van " + str(SCORE_DOEL) + " bereikt in generatie " + str(gen) + "!")

    print("\nBest genome:\n{!s}".format(winner))

    # Het beste vogeltje bewaren met pickle
    with open(WINNER_BESTAND, "wb") as f:
        pickle.dump(winner, f)
    print("Winnaar opgeslagen in " + WINNER_BESTAND)

    return winner


def speel_met_beste_vogel(config_file, genome=None):
    """
    Eindspel: alleen het beste vogeltje speelt, op normale snelheid, zonder
    scorelimiet. Het genome wordt uit winner.pkl geladen als er geen wordt meegegeven.
    Het spel stopt als de vogel doodgaat of het venster wordt gesloten.
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    if genome is None:
        with open(WINNER_BESTAND, "rb") as f:
            genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    bird = Bird(BIRD_X, BIRD_START_Y)
    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0
    clock = pygame.time.Clock()

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pipe_ind = dichtstbijzijnde_pipe(pipes)

        bird.move()
        output = net.activate(netwerk_invoer(bird, pipes[pipe_ind]))
        if output[0] > 0.5:
            bird.jump()

        base.move()

        add_pipe = False
        rem = []
        geraakt = False
        for pipe in pipes:
            pipe.move()
            if pipe.collide(bird):
                geraakt = True
            if pipe.x + pipe.breedte < 0:
                rem.append(pipe)
            if not pipe.passed and pipe.x < BIRD_X:
                pipe.passed = True
                add_pipe = True

        if add_pipe:
            score += 1
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        if geraakt or bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
            break

        draw_window(WIN, [bird], pipes, base, score, 0, pipe_ind, toon_generatie=False)

    print("Eindspel afgelopen met score: " + str(score))
    eind_label = END_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    WIN.blit(eind_label, (WIN_WIDTH / 2 - eind_label.get_width() / 2, 350))
    pygame.display.update()
    pygame.time.wait(3000)


if __name__ == '__main__':
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # Pad naar de config bepalen zodat het script overal vandaan werkt
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, 'config-feedforward.txt')

    if len(sys.argv) > 1 and sys.argv[1] == "speel":
        TOON_SCHERM = True
        speel_met_beste_vogel(config_path)
    else:
        winner = run(config_path)
        # Na het trainen altijd met scherm het eindspel tonen
        TOON_SCHERM = True
        speel_met_beste_vogel(config_path, winner)

    pygame.quit()
