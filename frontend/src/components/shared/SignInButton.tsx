"use client";

import React from "react";
import Link from "next/link";

type Props = { text?: string };

const SignInButton = ({ text = "Sign Up" }: Props) => {
  return (
    <Link href="/signin">
      <button className="flex h-fit min-h-[33.6px] min-w-20 items-center justify-center rounded-xl border bg-neutral-200 px-3 py-1.5 text-sm font-semibold text-black shadow-[inset_0px_1.2px_0px_#ffffff] transition-all duration-300 hover:bg-neutral-300">
        {text}
      </button>
    </Link>
  );
};

export default SignInButton;
