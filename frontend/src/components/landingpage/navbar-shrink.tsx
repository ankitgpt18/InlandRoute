"use client";

import { Menu, X } from "lucide-react";
import Link from "next/link";
import React, { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { navItems } from "@/constants";
import Image from "next/image";
import SignInButton from "../shared/SignInButton";
import AnchorNav from "../shared/anchor-nav";

const Navbar = () => {
  const [menuState, setMenuState] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <header>
      <nav
        data-state={menuState && "active"}
        className="group fixed z-50 w-full px-2"
      >
        <div
          className={cn(
            "mx-auto mt-2 max-w-[95rem] border border-neutral-800/5 px-6 transition-all duration-300 lg:px-12",
            isScrolled &&
              "max-w-7xl rounded-2xl border border-neutral-800/50 bg-neutral-900/50 backdrop-blur-lg lg:px-5",
          )}
        >
          <div className="relative flex flex-wrap items-center justify-between gap-6 py-1 lg:gap-0">
            <div className="flex w-full justify-between items-center lg:w-auto">
              <Link href="/" className="flex items-center gap-2">
                <Image
                  width={500}
                  height={500}
                  src={"/logo.png"}
                  alt="InlandRoute"
                  quality={100}
                  priority={true}
                  className="mt-1 h-10 w-10 flex-shrink-0 rounded-xl object-cover"
                />
                <h3 className="text-xl font-bold text-white">InlandRoute</h3>
              </Link>

              <button
                onClick={() => setMenuState(!menuState)}
                aria-label={menuState == true ? "Close Menu" : "Open Menu"}
                className="relative z-20 -m-2.5 -mr-4 block cursor-pointer p-2.5 text-white lg:hidden"
              >
                <Menu className="m-auto size-6 duration-200 group-data-[state=active]:rotate-180 group-data-[state=active]:scale-0 group-data-[state=active]:opacity-0" />
                <X className="absolute inset-0 m-auto size-6 -rotate-180 scale-0 opacity-0 duration-200 group-data-[state=active]:rotate-0 group-data-[state=active]:scale-100 group-data-[state=active]:opacity-100" />
              </button>
            </div>

            <div className="absolute inset-0 m-auto hidden size-fit lg:block">
              <ul className="flex gap-8 text-sm">
                {navItems.map((item) => (
                  <AnchorNav
                    key={item.name + item.href}
                    absolute
                    href={item.href as any}
                  >
                    {item.name}
                  </AnchorNav>
                ))}
              </ul>
            </div>

            <div className="mb-6 hidden w-full flex-wrap items-center justify-end space-y-8 rounded-3xl border border-neutral-800 bg-neutral-900 p-6 shadow-none shadow-zinc-300/20 group-data-[state=active]:block md:flex-nowrap lg:m-0 lg:flex lg:w-fit lg:gap-6 lg:space-y-0 lg:border-transparent lg:bg-transparent lg:p-0 lg:shadow-none lg:group-data-[state=active]:flex">
              <div className="lg:hidden">
                <ul className="space-y-6 text-base">
                  {navItems.map((item) => (
                    <AnchorNav
                      key={item.name + item.href}
                      absolute
                      href={item.href as any}
                    >
                      {item.name}
                    </AnchorNav>
                  ))}
                </ul>
              </div>
              <div className="flex flex-col items-center space-y-3 sm:flex-row sm:gap-3 sm:space-y-0 md:w-fit w-full">
                <SignInButton text={"Sign Up"} />
              </div>
            </div>
          </div>
        </div>
      </nav>
    </header>
  );
};

export default Navbar;
