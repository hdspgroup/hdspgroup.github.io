export type PersonLink = {
  label: string;
  href: string;
  icon: string;
};

export type Person = {
  name: string;
  title: string;
  description?: string;
  links?: PersonLink[];
};

export type PeopleCategory = {
  title: string;
  description: string;
  icon: string;
  members: Person[];
};

type PersonSeed = Omit<Person, 'links'> & {
  scholarHref?: string;
  links?: PersonLink[];
};

const scholarPlaceholder = 'https://scholar.google.com/';

const scholarLink = (href: string = scholarPlaceholder): PersonLink => ({
  label: 'Scholar',
  href,
  icon: 'tabler:school',
});

const buildPerson = ({ scholarHref = scholarPlaceholder, links = [], ...person }: PersonSeed): Person => ({
  ...person,
  links: [scholarLink(scholarHref), ...links],
});

export const principalInvestigator: Person & { interests: string[]; image: string } = {
  ...buildPerson({
    name: 'Professor Henry Arguello Fuentes',
    title:
      'Principal Investigator | Ph.D. in Electrical and Computer Engineering | Associate Professor, Universidad Industrial de Santander',
    description:
      'Professor Henry Arguello leads the HDSP Group, focusing on high-dimensional signal processing, compressed sensing, and computational imaging.',
    scholarHref: 'https://scholar.google.com/citations?user=R7gjbGIAAAAJ&hl=en',
    links: [
      {
        label: 'LinkedIn',
        href: 'https://www.linkedin.com/in/henry-arguello-2905929/',
        icon: 'tabler:brand-linkedin',
      },
    ],
  }),
  image: '~/assets/images/prof-henry-arguello.jpeg',
  interests: [
    'Statistical signal processing',
    'Super-resolution',
    'Inverse problems',
    'Optical imaging',
    'Video processing',
    'Hyperspectral imaging',
    'Compressive sensing',
  ],
};

export const peopleCategories: PeopleCategory[] = [
  {
    title: 'Professors',
    description: 'Faculty members contributing to the academic and scientific leadership of the group.',
    icon: 'tabler:chalkboard',
    members: [buildPerson({ name: 'Hans Garcia', title: 'Ph.D. Professor' })],
  },
  {
    title: 'Doctoral Students',
    description: 'Ph.D. researchers advancing core HDSP research lines and multidisciplinary projects.',
    icon: 'tabler:microscope',
    members: [
      buildPerson({ name: 'Paul Goyes', title: 'Ph.D. in Computer Science' }),
      buildPerson({ name: 'Jhon Lopez', title: 'Ph.D. in Computer Science' }),
      buildPerson({ name: 'Kevin Arias', title: 'Ph.D. in Computer Science' }),
      buildPerson({ name: 'Emmanuel Martinez', title: 'Ph.D. in Computer Science' }),
      buildPerson({ name: 'Leon Suarez', title: 'Ph.D. in Computer Science' }),
      buildPerson({ name: 'Juan Carlos Vega', title: 'Ph.D. in Computer Science' }),
      buildPerson({ name: 'Sergio Urrea', title: 'Ph.D. in Engineering' }),
      buildPerson({ name: 'Pablo Gomez', title: 'Ph.D. in Engineering' }),
      buildPerson({ name: 'Roman Jacome', title: 'Ph.D. in Engineering' }),
      buildPerson({
        name: 'Karen Fonseca',
        title: 'Ph.D. in Engineering',
        scholarHref: 'https://scholar.google.com/citations?user=KQCZTqAAAAAJ&hl=en',
      }),
    ],
  },
  {
    title: 'Master Students',
    description: 'Graduate students exploring applications across systems, electronics, and geophysics.',
    icon: 'tabler:atom-2',
    members: [
      buildPerson({ name: 'Javier Torres', title: 'M.Sc. in Systems and Computer Engineering' }),
      buildPerson({ name: 'Sebastian Ardila', title: 'M.Sc. in Electronic Engineering' }),
      buildPerson({ name: 'Ana Mantilla', title: 'M.Sc. in Geophysics' }),
    ],
  },
  {
    title: 'Undergraduate Students',
    description: 'Student researchers developing projects, prototypes, and scientific training within HDSP.',
    icon: 'tabler:users-group',
    members: [
      buildPerson({
        name: 'Laura C. Diaz-Delgado',
        title: 'Computer Science Engineering',
        scholarHref: 'https://scholar.google.com/citations?user=jAfnVpoAAAAJ&hl=en',
      }),
      buildPerson({ name: 'Julio Gutierrez', title: 'Computer Science Engineering' }),
      buildPerson({ name: 'Jose Barrios', title: 'Electronic Engineering' }),
      buildPerson({ name: 'Lamar Rivera', title: 'Physics' }),
      buildPerson({ name: 'Santiago Rodriguez', title: 'Electronic Engineering' }),
      buildPerson({ name: 'Carlos Mogollon', title: 'Electronic Engineering' }),
      buildPerson({ name: 'Nohelia Agudelo', title: 'Electronic Engineering' }),
      buildPerson({ name: 'Javier Quiroga', title: 'Electronic Engineering' }),
      buildPerson({ name: 'Ernesto Vasquez', title: 'Electronic Engineering' }),
      buildPerson({ name: 'Deisy Camacho', title: 'Mathematics' }),
      buildPerson({ name: 'Daniel Diaz', title: 'Electronic Engineering' }),
      buildPerson({ name: 'Juan Diego Cardenas', title: 'Computer Science Engineering' }),
    ],
  },
  {
    title: 'Administrative and Professionals',
    description: 'Management and professional support that sustains the group\'s projects and operations.',
    icon: 'tabler:briefcase',
    members: [
      buildPerson({ name: 'Ana Gutierrez', title: 'Master\'s in Project Management' }),
      buildPerson({ name: 'Marcela Rincon', title: 'Industrial Engineer | Specialist in Strategic Management' }),
    ],
  },
  {
    title: 'Collaborators',
    description: 'Researchers and academic partners linked to HDSP projects and scientific exchange.',
    icon: 'tabler:world',
    members: [
      buildPerson({ name: 'Said Pertuz', title: 'Ph.D. Professor' }),
      buildPerson({ name: 'Sergio Castillo', title: 'Ph.D. Professor' }),
      buildPerson({ name: 'Hoover Rueda', title: 'Ph.D. Professor' }),
      buildPerson({ name: 'Luis Gonzalez', title: 'M.Sc. Professor' }),
      buildPerson({ name: 'Laura Galvis', title: 'Ph.D. Professor' }),
      buildPerson({ name: 'Jorge Bacca', title: 'Ph.D. Professor' }),
      buildPerson({ name: 'Alejandra Hernandez', title: 'M.Sc. in Geophysics' }),
      buildPerson({ name: 'Ofelia Villarreal', title: 'M.Sc. in Electronic Engineering' }),
      buildPerson({ name: 'Romario Gualdron', title: 'M.Sc. in Computer Science Engineering' }),
      buildPerson({ name: 'Paula Arguello', title: 'Ph.D. student in Computer Science' }),
    ],
  },
];
